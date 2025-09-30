# Working with Arrays

Arrays are an important construct of any programming language. To keep data of a similar type together, we need arrays. Arrays are heavily used in applications where elements have to be accessed at random. Arrays are also a prime choice when you need to sort elements, look for desired data in a collection, and find common or unique data between two sets. Arrays are assigned contiguous memory locations and are a very popular structure for sorting and searching data collections because any element of an array can be accessed by simply specifying its subscript or index location. This chapter will cover recipes that include operations commonly applied to arrays.

In this chapter, we will learn how to make the following recipes using arrays:

*   Inserting an element into a one-dimensional array
*   Multiplying two matrices
*   Finding the common elements in two arrays
*   Finding the difference between two sets or arrays
*   Finding the unique elements in an array
*   Finding whether a matrix is sparse
*   Merging two sorted arrays into one

Let's begin with the first recipe!

# Inserting an element in an array

In this recipe, we will learn how to insert an element in-between an array. You can define the length of the array and also specify the location where you want the new value to be inserted. The program will display the array after the value has been inserted.

# How to do it…

1\. Let's assume that there is an array, **p**, with five elements, as follows:

![](img/eb343648-1bfa-4181-a25f-caa6bc3fc4db.png)

Figure 1.1

Now, suppose you want to enter a value, say **99**, at the third position. We will write a C program that will give the following output:

![](img/03231163-5321-4c3a-ac8d-958579ad1f4f.png)

Figure 1.2

Here are the steps to follow to insert an element in an array:

1.  Define a macro called `max` and initialize it to a value of `100`:

```cpp
#define max 100
```

2.  Define an array `p` of size max elements:

```cpp
int p[max]
```

3.  Enter the length of the array when prompted. The length you enter will be assigned to a variable `n`:

```cpp
printf("Enter length of array:");
scanf("%d",&n);
```

4.  A `for` loop will be executed prompting you to enter the elements of the array:

```cpp
for(i=0;i<=n-1;i++ )
    scanf("%d",&p[i]);
```

5.  Specify the position in the array where the new value has to be inserted:

```cpp
printf("\nEnter position where to insert:");
scanf("%d",&k);
```

6.  Because the arrays in C are zero-based, the position you enter is decremented by 1:

```cpp
k--;
```

7.  To create space for the new element at the specified index location, all the elements are shifted one position down:

```cpp
for(j=n-1;j>=k;j--)
    p[j+1]=p[j];
```

8.  Enter the new value which will be inserted at the vacated index location:

```cpp
printf("\nEnter the value to insert:");
scanf("%d",&p[k]);
```

Here is the `insertintoarray.c` program for inserting an element in between an array:

```cpp
#include<stdio.h>
#define max 100
void main()
{
    int p[max], n,i,k,j;
    printf("Enter length of array:");
    scanf("%d",&n);
    printf("Enter %d elements of array\n",n);
    for(i=0;i<=n-1;i++ )
        scanf("%d",&p[i]);
    printf("\nThe array is:\n");
    for(i = 0;i<=n-1;i++)
        printf("%d\n",p[i]);
    printf("\nEnter position where to insert:");
    scanf("%d",&k);
    k--;/*The position is always one value higher than the subscript, so it is decremented by one*/             
    for(j=n-1;j>=k;j--)
        p[j+1]=p[j];
    /* Shifting all the elements of the array one position down from the location of insertion */
    printf("\nEnter the value to insert:");
    scanf("%d",&p[k]);
    printf("\nArray after insertion of element: \n");
    for(i=0;i<=n;i++)
        printf("%d\n",p[i]);
}

```

Now, let's go behind the scenes to understand the code better.

# How it works...

Because we want to specify the length of the array, we will first define a macro called `max` and initialize it to a value of 100\. I have defined the value of max as 100 because I assume that I will not need to enter more than 100 values in an array, but it can be any value as desired. An array, `p`, is defined of size `max` elements. You will be prompted to specify the length of the array. Let's specify the length of the array as 5\. We will assign the value `5` to the variable `n`. Using a `for` loop, you will be asked to enter the elements of the array.

Let's say you enter the values in the array, as shown in *Figure 1.1* given earlier:

![](img/3e09ac0d-757d-4d8c-a972-797bf2a62bfb.png)

In the preceding diagram, the numbers, 0, 1, 2, and so on are known as index or subscript and are used for assigning and retrieving values from an array. Next, you will be asked to specify the position in the array where the new value has to be inserted. Suppose, you enter `3`, which is assigned to the variable `k`. This means that you want to insert a new value at location 3 in the array.

Because the arrays in C are zero-based, position 3 means that you want to insert a new value at index location 2, which is **p[2]**. Hence, the position entered in `k` is decremented by 1.

To create space for the new element at index location **p[2]**, all the elements are shifted one position down. This means that the element at **p[4]** is moved to index location **p[5]**, the one at **p[3]** is moved to **p[4]**, and the element at **p[2]** is moved to **p[3]**, as follows:

![](img/3ab3e7ea-0a38-4d83-8a16-3e6aef8abadb.png)

Figure 1.3

Once the element from the target index location is safely copied to the next location, you will be asked to enter the new value. Suppose you enter the new value as `99`; that value will be inserted at index location **p[2]**, as shown in *Figure 1.2,* given earlier:

![](img/7e4bfb32-93c9-4fd9-be3c-d937eef6de99.png)

Let’s use GCC to compile the `insertintoarray.c` program, as shown in this statement:

```cpp
D:\CBook>gcc insertintoarray.c -o insertintoarray
```

Now, let’s run the generated executable file, `insertintoarray.exe`, to see the program output:

```cpp
D:\CBook>./insertintoarray
Enter length of array:5
Enter 5 elements of array
10
20
30
40
50

The array is:
10
20
30
40
50

Enter target position to insert:3
Enter the value to insert:99
Array after insertion of element:
10
20
99
30
40
50
```

Voilà! We've successfully inserted an element in an array. 

# There's more...

What if we want to delete an element from an array? The procedure is simply the reverse; in other words, all the elements from the bottom of the array will be copied one place up to replace the element that was deleted.

Let's assume array **p** has the following five elements (*Figure 1.1*):

![](img/f3101b54-16c1-447f-87df-2a9bb040afdf.png)

Suppose, we want to delete the third element, in other words, the one at **p[2]**, from this array. To do so, the element at **p[3]** will be copied to **p[2]**, the element at **p[4]** will be copied to **p[3]**, and the last element, which here is at **p[4]**, will stay as it is:

![](img/837308f4-9834-44b3-b6fc-4af5d74880c3.png)

Figure 1.4

The `deletefromarray.c` program for deleting the array is as follows:

```cpp
#include<stdio.h>
void main()
{
    int p[100],i,n,a;
    printf("Enter the length of the array: ");
    scanf("%d",&n);
    printf("Enter %d elements of the array \n",n);
    for(i=0;i<=n-1;i++)
        scanf("%d",&p[i]);
    printf("\nThe array is:\n");\
    for(i=0;i<=n-1;i++)
        printf("%d\n",p[i]);
    printf("Enter the position/location to delete: ");
    scanf("%d",&a);
    a--;
    for(i=a;i<=n-2;i++)
    {
        p[i]=p[i+1];
        /* All values from the bottom of the array are shifted up till 
        the location of the element to be deleted */
    }
    p[n-1]=0;
    /* The vacant position created at the bottom of the array is set to 
    0 */
    printf("Array after deleting the element is\n");
    for(i=0;i<= n-2;i++)
        printf("%d\n",p[i]);
}
```

Now, let's move on to the next recipe!

# Multiplying two matrices

A prerequisite for multiplying two matrices is that the number of columns in the first matrix must be equal to the number of rows in the second matrix.

# How to do it…

1.  Create two matrices of orders **2 x 3** and **3 x 4** each. 
2.  Before we make the matrix multiplication program, we need to understand how matrix multiplication is performed manually. To do so, let's assume that the two matrices to be multiplied have the following elements:

![](img/a9026553-5394-4d74-9602-8c3ddad23829.png)

Figure 1.5

3.  The resultant matrix will be of the order **2 x 4**, that is, the resultant matrix will have the same number of rows as the first matrix and the same number of columns as the second matrix:

![](img/3b397ad3-905e-44fd-9455-25eacaa28915.png)

Figure 1.6

Essentially, the resultant matrix of the order **2 x 4** will have the following elements:

![](img/30777e6a-c99a-4621-b57b-8c77a7636dfe.png)

Figure 1.7

4.  The element **first row, first column** in the resultant matrix is computed using the following formula:

<q>SUM(first element of the first row of the first matrix × first element of the first column of the second matrix), (second element of the first row... × second element of the first column...), (and so on...)</q>

For example, let's assume the elements of the two matrices are as shown in *Figure 1.5*.  The elements in the first row and the first column of the resultant matrix will be computed as follows:

![](img/927c04ce-0fc1-4a49-889e-8763a446f32f.png)

Figure 1.8

5.  Hence, the element in **first row, first column** in the resultant matrix will be as follows:

**(3×6)+(9×3)+(7×5)**
**=18 + 27 + 35**
**=80**

*Figure 1.9* explains how the rest of the elements are computed in the resultant matrix:

![](img/0d797c41-8b82-4911-80a4-71d82b3c2f91.png)

Figure 1.9

The `matrixmulti.c` program for multiplying the two matrices is as follows:

```cpp
#include  <stdio.h>
int main()
{
  int matA[2][3], matB[3][4], matR[2][4];
  int i,j,k;
  printf("Enter elements of the first matrix of order 2 x 3 \n");
  for(i=0;i<2;i++)
  {
    for(j=0;j<3;j++)
    {
      scanf("%d",&matA[i][j]);
    }
  }
  printf("Enter elements of the second matrix of order 3 x 4 \n");
  for(i=0;i<3;i++)
  {
    for(j=0;j<4;j++)
    {
      scanf("%d",&matB[i][j]);
    }
  }
  for(i=0;i<2;i++)
  {
    for(j=0;j<4;j++)
    {
      matR[i][j]=0;
      for(k=0;k<3;k++)
      {
        matR[i][j]=matR[i][j]+matA[i][k]*matB[k][j];
      }
    }
  }
  printf("\nFirst Matrix is \n");
  for(i=0;i<2;i++)
  {
    for(j=0;j<3;j++)
    {
      printf("%d\t",matA[i][j]);
    }
    printf("\n");
  }
  printf("\nSecond Matrix is \n");
  for(i=0;i<3;i++)
  {
    for(j=0;j<4;j++)
    {
      printf("%d\t",matB[i][j]);
    }
    printf("\n");
  }
  printf("\nMatrix multiplication is \n");
  for(i=0;i<2;i++)
  {
    for(j=0;j<4;j++)
    {
      printf("%d\t",matR[i][j]);
    }
    printf("\n");
  }
  return 0;
}
```

Now, let's go behind the scenes to understand the code better.

# How it works...

The two matrices are defined `matA` and `matB` of the orders 2 x 3 and 3 x 4, respectively, using the following statement:

```cpp
int matA[2][3], matB[3][4]
```

You will be asked to enter the elements of the two matrices using the nested `for` loops. The elements in the matrix are entered in row-major order, in other words, all the elements of the first row are entered first, followed by all the elements of the second row, and so on.

In the nested loops, `for i` and `for j`, the outer loop, `for i`, represents the row and the inner loop, and `for j` represents the column.

While entering the elements of matrices `matA` and `matB`, the values entered in the two matrices will be assigned to the respective index locations of the two-dimensional arrays as follows:

![](img/e81eb6e0-0021-4920-b05f-260cdae9576f.png)

Figure 1.10

The nested loops that actually compute the matrix multiplication are as follows:

```cpp
  for(i=0;i<2;i++)
  {
    for(j=0;j<4;j++)
    {
      matR[i][j]=0;
      for(k=0;k<3;k++)
      {
        matR[i][j]=matR[i][j]+matA[i][k]*matB[k][j];
      }
    }
  }
```

The variable `i` represents the row of the resultant matrix, `j` represents the column of the resultant matrix, and `k` represents the common factor. The <q>common factor</q> here means the column of the first matrix and the row of the second matrix.

Recall that the prerequisite for matrix multiplication is that the column of the first matrix should have the same number of rows as the second matrix. Because the respective elements have to be added after multiplication, the element has to be initialized to `0` before addition.

The following statement initializes the elements of the resultant matrix:

```cpp
      matR[i][j]=0;
```

The `for k` loop inside the nested loops helps in selecting the elements in the rows of the first matrix and multiplying them by elements of the column of the second matrix:

```cpp
matR[i][j]=matR[i][j]+matA[i][k]*matB[k][j];
```

Let's use GCC to compile the `matrixmulti.c` program as follows:

```cpp
D:\CBook>gcc matrixmulti.c -o matrixmulti
```

Let's run the generated executable file, `matrixmulti.exe`, to see the output of the program:

```cpp
D:\CBook\Chapters\1Arrays>./matrixmulti

Enter elements of the first matrix of order 2 x 3
3
9
7
1
5
4

Enter elements of the second matrix of order 3 x 4
6 2 8 1
3 9 4 0
5 3 1 3

First Matrix is
3 9 7 
1 5 4

Second Matrix is
6 2 8 1
3 9 4 0
5 3 1 3

Matrix multiplication is
80 108 67 24
41 59 32 13
```

Voilà! We've successfully multiplied two matrices.

# There’s more…

One thing that you might notice while entering the elements of the matrix is that there are two ways of doing it.

1.  The first method is that you press *Enter* after inputting each element as follows:

```cpp
3
9
7
1
5
4
```

The values will be automatically assigned to the matrix in row-major order, in other words, `3` will be assigned to `matA[0][0]`, `9` will be assigned to `matA[0][1]`, and so on.

2.  The second method of entering elements in the matrix is as follows:

```cpp
6 2 8 1
3 9 4 0
5 3 1 3
```

Here, `6` will be assigned to `matB[0][0]`, `2` will be assigned to `matB[0][1]`, and so on.

Now, let's move on to the next recipe!

# Finding the common elements in two arrays

Finding the common elements in two arrays is akin to finding the intersection of two sets. Let's learn how to do it.

# How to do it…

1.  Define two arrays of a certain size and assign elements of your choice to both the arrays. Let's assume that we created two arrays called **p** and **q**, both of size four elements:

![](img/976f5af6-1cd0-4e88-a37b-5c9d0ad7dec3.png)

Figure 1.11

2.  Define one more array. Let's call it array **r**, to be used for storing the elements that are common between the two arrays.
3.  If an element in array **p** exists in the array **q**, it is added to array **r**. For instance, if the element at the first location in array **p**, which is at **p[0]**, does not appear in array **q**, it is discarded, and the next element, at **p[1]**, is picked up for comparison.

4.  And if the element at **p[0]** is found anywhere in array **q**, it is added to array **r**, as follows:

![](img/470ee921-9ded-42c6-b65f-b77af62624da.png)

Figure 1.12

5.  This procedure is repeated with other elements of array **q**. That is, **p[1]** is compared with **q[0]**, **q[1]**, **q[2]**, and **q[3]**. If **p[1]** is not found in array **q**, then before inserting it straightaway into array **r**, it is compared with the existing elements of array **r** to avoid repetitive elements.
6.  Because the element at **p[1]** appears in array **q** and is not already present in array **r**, it is added to array **r** as follows:

![](img/c321afb9-d7f5-4898-9b9f-804cb520f691.png)

Figure 1.13

The `commoninarray.c` program for establishing common elements among the two arrays is as follows:

```cpp
#include<stdio.h>
#define max 100

int ifexists(int z[], int u, int v)
{
    int i;
    if (u==0) return 0;
    for (i=0; i<=u;i++)
        if (z[i]==v) return (1);
    return (0);
}
void main()
{
    int p[max], q[max], r[max];
    int m,n;
    int i,j,k;
    k=0;
    printf("Enter the length of the first array:");
    scanf("%d",&m);
    printf("Enter %d elements of the first array\n",m);
    for(i=0;i<m;i++ )
        scanf("%d",&p[i]);
    printf("\nEnter the length of the second array:");
    scanf("%d",&n);
    printf("Enter %d elements of the second array\n",n);
    for(i=0;i<n;i++ )
        scanf("%d",&q[i]);
    k=0;
    for (i=0;i<m;i++)
    {
        for (j=0;j<n;j++)
        {
           if (p[i]==q[j])
           {
               if(!ifexists(r,k,p[i]))
               {
                   r[k]=p[i];
                   k++;
               }
            }
        }
    }
    if(k>0)
    {
        printf("\nThe common elements in the two arrays are:\n");
        for(i = 0;i<k;i++)
            printf("%d\n",r[i]);
    }
    else
        printf("There are no common elements in the two arrays\n");
}
```

Now, let's go behind the scenes to understand the code better.

# How it works...

A macro, `max`, is defined of size `100`. A function, `ifexists()`, is defined that simply returns `true (1)` or `false (0)`. The function returns `true` if the supplied value exists in the specified array, and `false` if it doesn't.

Two arrays are defined, called `p` and `q`, of size `max` (in other words, 100 elements). You will be prompted to specify the length of the array, `p`, and then asked to enter the elements in that array. After that, you will be asked to specify the length of array `q`, followed by entering the elements in array `q`.

Thereafter, `p[0]`, the first element in array `p` , is picked up, and by using the `for` loop, `p[0]` is compared with all the elements of array `q`. If `p[0]` is found in array `q`, then `p[0]` is added to the resulting array, `r`.

After a comparison of `p[0]`, the second element in array p, `p[1]`, is picked up and compared with all the elements of array `q`. The procedure is repeated until all the elements of array `p` are compared with all the elements of array `q`.

If any elements of array `p` are found in array `q`, then before adding that element to the resulting array, `r`, it is run through the `ifexists()` function to ensure that the element does not already exist in array `r`. This is because we don't want repetitive elements in array `r`.

Finally, all the elements in array `r`, which are the common elements of the two arrays, are displayed on the screen.

Let's use GCC to compile the `commoninarray.c` program as follows:

```cpp
D:\CBook>gcc commoninarray.c -o commoninarray
```

Now, let's run the generated executable file, `commoninarray.exe`, to see the output of the program:

```cpp
D:\CBook>./commoninarray
Enter the length of the first array:5
Enter 5 elements in the first array
1
2
3
4
5

Enter the length of the second array:4
Enter 4 elements in the second array
7
8
9
0

There are no common elements in the two arrays
```

Because there were no common elements between the two arrays entered previously, we can't quite say that we've truly tested the program. Let's run the program again, and this time, we will enter the array elements such that they have something in common.

```cpp
D:\CBook>./commoninarray
Enter the length of the first array:4
Enter 4 elements in the first array
1
2
3
4

Enter the length of the second array:4
Enter 4 elements in the second array
1
4
1
2

The common elements in the two arrays are:
1
2
4
```

Voilà! We've successfully identified the common elements between two arrays.

# Finding the difference between two sets or arrays

When we talk about the difference between two sets or arrays, we are referring to all the elements of the first array that don't appear in the second array. In essence, all the elements in the first array that are not common to the second array are referred to as the difference between the two sets. The difference in sets `p` and `q`, for example, will be denoted by `p – q`.

If array `p`, for example, has the elements `{1, 2, 3, 4}`, and array `q` has the elements `{2, 4, 5, 6}`, then the difference between the two arrays, `p - q`, will be  `{1,3}`.  Let's find out how this is done.

# How to do it…

1.  Define two arrays, say `p` and `q`, of a certain size and assign elements of your choice to both the arrays.
2.  Define one more array, say `r`, to be used for storing the elements that represent the difference between the two arrays.
3.  Pick one element from array `p` and compare it with all the elements of the array `q`.
4.  If the element of array `p` exists in array `q`, discard that element and pick up the next element of array `p` and repeat from step 3.
5.  If the element of array `p` does not exist in array `q`, add that element in array `r`. Before adding that element to array `r`, ensure that it does not already exist in array `r`.
6.  Repeat steps 3 to 5 until all the elements of array `p` are compared.
7.  Display all the elements in array `r`, as these are the elements that represent the difference between arrays `p` and `q`.

The `differencearray.c` program to establish the difference between two arrays is as follows:

```cpp
#include<stdio.h>
#define max 100

int ifexists(int z[], int u, int v)
{
    int i;
    if (u==0) return 0;
    for (i=0; i<=u;i++)
        if (z[i]==v) return (1);
    return (0);
}

void main()
{
    int p[max], q[max], r[max];
    int m,n;
    int i,j,k;
    printf("Enter length of first array:");
    scanf("%d",&m);
    printf("Enter %d elements of first array\n",m);
    for(i=0;i<m;i++ )
        scanf("%d",&p[i]);
    printf("\nEnter length of second array:");
    scanf("%d",&n);
    printf("Enter %d elements of second array\n",n);
    for(i=0;i<n;i++ )                                                                                    scanf("%d",&q[i]);
    k=0;
    for (i=0;i<m;i++)               
    {                                
        for (j=0;j<n;j++)                                
        {
            if (p[i]==q[j])
            {                                                                                                                                    break;                                                   
            }
        }
        if(j==n)
        {
            if(!ifexists(r,k,p[i]))                                               
            {
                r[k]=p[i];
                k++;
            }
        }
    }
    printf("\nThe difference of the two array is:\n");
    for(i = 0;i<k;i++)
        printf("%d\n",r[i]);
}
```

Now, let's go behind the scenes to understand the code better.

# How it works...

We defined two arrays called **p** and **q**. We don't want to fix the length of these arrays, so we should define a macro called `max` of value `100` and set the two arrays, **p** and **q**, to the size of `max`.

Thereafter, you will be prompted to specify the size of the first array and enter the elements in the first array, **p**. Similarly, you will be asked to specify the length of the second array, **q**, followed by entering the elements in the second array.

Let's assume you have specified the length of both arrays as 4 and have entered the following elements:

![](img/87a30378-73c8-4c26-8a32-a8b81f1a7983.png)

Figure 1.14

We need to pick up one element at a time from the first array and compare it with all the elements of the second array. If an element in array **p** does not appear in array **q**, it will be assigned to the third array we created, array **r**.

Array **r** will be used for storing the elements that define the difference between two arrays. As shown in *Figure 1.15*, the first element of array **p**, in other words, at **p[0]**, is compared with all the elements of array **q**, in other words, with **q[0]**, **q[1]**, **q[2]**, and **q[3]**.

Because the element at **p[0]**, which is **1**, does not appear in array **q**, it will be added to the array **r**, indicating the first element representing the difference between the two arrays:

![](img/a4f3b450-91a8-4f84-87da-4ce21104abff.png)

Figure 1.15

Because the element at **p[1]**, which is **2**, appears in array **q**, it is discarded, and the next element in array **p**, in other words, **p[2]**, is picked up and compared with all the elements in array **q**.

As the element at **p[2]** does not appear in array **q**, it is added to array **r** at the next available location, which is **r[1]** (see *Figure 1.16* as follows):

![](img/4e03ad94-446e-4f8a-a653-35e8f27cbe50.png)

Figure 1.16

Continue the procedure until all the elements of array **p** are compared with all the elements of array **q**. Finally, we will have array **r**, with the elements showing the difference between our two arrays, **p** and **q**.

Let's use GCC to compile our program, `differencearray.c`, as follows:

```cpp
D:\CBook>gcc differencearray.c -o differencearray
```

Now, let's run the generated executable file, `differencearray`, to see the output of the program:

```cpp
D:\CBook>./differencearray
Enter length of first array:4
Enter 4 elements of first array
1
2
3
4
Enter length of second array:4
Enter 4 elements of second array
2
4
5
6
The difference of the two array is:
1
3
```

Voilà! We've successfully found the difference between two arrays. Now, let's move on to the next recipe!

# Finding the unique elements in an array

In this recipe, we will learn how to find the unique elements in an array, such that the repetitive elements in the array will be displayed only once.

# How to do it… 

1.  Define two arrays, **p** and **q**, of a certain size and assign elements only to array **p**. We will leave array **q** blank.
2.  These will be our source and target arrays, respectively. The target array will contain the resulting unique elements of the source array.

3.  After that, each of the elements in the source array will be compared with the existing elements in the target array. 
4.  If the element in the source array exists in the target array, then that element is discarded and the next element in the source array is picked up for comparison.
5.  If the source array element does not exist in the target array, it is copied into the target array.
6.  Let's assume that array **p** contains the following repetitive elements:

![](img/db77e611-7d11-4bab-a6e9-8f4cef21e955.png)

Figure 1.17

7.  We will start by copying the first element of the source array, **p**, into the target array, **q**, in other words, **p[0]** into array **q[0]**, as follows:

![](img/14b8037f-0e33-4759-a70b-3e724f91659d.png)

Figure 1.18

8.  Next, the second array element of **p**, in other words, **p[1]**, is compared with all the existing elements of array **q**. That is, **p[1]** is compared with array **q** to check whether it already exists in array **q**, as follows:

![](img/9d99f9a0-e92a-469a-9a79-5e6b27cf9b0e.png)

Figure 1.19

9.  Because **p[1]** does not exist in array **q**, it is copied at **q[1]**, as shown in *Figure 1.20*:

![](img/cc8405c7-8f44-4f6c-90d7-c6409d8414fe.png)

Figure 1.20

10.  This procedure is repeated until all the elements of array **p** are compared with array q. In the end, we will have array **q**, which will contain the unique elements of array **p**.

Here is the `uniqueelements.c` program for finding the unique elements in the first array:

```cpp
#include<stdio.h>
#define max 100

int ifexists(int z[], int u, int v)
{
    int i;
    for (i=0; i<u;i++)
        if (z[i]==v) return (1);
    return (0);
}

void main()
{
    int p[max], q[max];
    int m;
    int i,k;
    k=0;
    printf("Enter length of the array:");
    scanf("%d",&m);
    printf("Enter %d elements of the array\n",m);
    for(i=0;i<m;i++ )
        scanf("%d",&p[i]);
    q[0]=p[0];
    k=1;
    for (i=1;i<m;i++)
    {
        if(!ifexists(q,k,p[i]))
        {
            q[k]=p[i];
            k++;
        }
    }
    printf("\nThe unique elements in the array are:\n");
    for(i = 0;i<k;i++)
        printf("%d\n",q[i]);
}
```

Now, let's go behind the scenes to understand the code better.

# How it works...

We will define a macro called `max` of size `100`. Two arrays, `p` and `q`, are defined of size `max`. Array `p` will contain the original elements, and array `q` will contain the unique elements of array `p`. You will be prompted to enter the length of the array and, thereafter, using the `for` loop, the elements of the array will be accepted and assigned to array `p`.

The following statement will assign the first element of array `p` to the first index location of our blank array, which we will name array `q`:

```cpp
q[0]=p[0]
```

A `for` loop is again used to access the rest of the elements of array `p`, one by one. First, the foremost element of array `p`, which is at `p[0]`, is copied to array `q` at `q[0]`.

Next, the second array `p` element, `p[1]`, is compared with all the existing elements of array `q`. That is, `p[1]` is checked against array `q` to confirm whether it is already present there.

Because there is only a single element in array `q`,  `p[1]` is compared with `q[0]`. Because `p[1]` does not exist in array `q`, it is copied at `q[1]`.

This procedure is repeated for all elements in array `p`. Each of the accessed elements of array `p` is run through the `ifexists()` function to check whether any of them already exist in array `q`. 

The function returns `1` if an element in array `p` already exists in array `q`. In that case, the element in array `p` is discarded and the next array element is picked up for comparison.

In case the `ifexists()` function returns `0`, confirming that the element in array `p` does not exist in array `q`, the array `p` element is added to array `q` at the next available index/subscript location.

When all the elements of array `p` are checked and compared, array `q` will have only the unique elements of array `p`.

Let's use GCC to compile the `uniqueelements.c` program as follows:

```cpp
D:\CBook>gcc uniqueelements.c -o uniqueelements
```

Now, let's run the generated executable file, `uniqueelements.exe`, to see the output of the program:

```cpp
D:\CBook>./uniqueelements
Enter the length of the array:5
Enter 5 elements in the array
1
2
3
2
1

The unique elements in the array are:
1
2
3
```

Voilà! We've successfully identified the unique elements in an array. Now, let's move on to the next recipe!

# Finding whether a matrix is sparse

A matrix is considered sparse when it has more zero values than non-zero values (and dense when it has more non-zero values). In this recipe, we will learn how to find out whether the specified matrix is sparse.

# How to do it…

1.  First, specify the order of the matrix. Then, you will be prompted to enter the elements in the matrix. Let's assume that you specified the order of the matrix as 4 x 4\. After entering the elements in the matrix, it might appear like this:

![](img/7ea8a31d-f60e-4d4c-a9b5-7e8c377bfc60.png)

Figure 1.21

2.  Once the elements of the matrix are entered, count the number of zeros in it. A counter for this purpose is initialized to **0**. Using nested loops, each of the matrix elements is scanned and, upon finding any zero elements, the value of the counter is incremented by 1.

3.  Thereafter, the following formula is used for establishing whether the matrix is sparse.

*If counter > [(the number of rows x the number of columns)/2] =  Sparse Matrix*

4.  Depending on the result of the preceding formula, one of the following messages will be displayed on the screen as follows:

```cpp
The given matrix is a sparse matrix
```

or

```cpp
The given matrix is not a sparse matrix
```

The `sparsematrix.c` program for establishing whether the matrix is sparse is as follows:

```cpp
#include <stdio.h>
#define max 100

/*A sparse matrix has more zero elements than nonzero elements */
void main ()
{
    static int arr[max][max];
    int i,j,r,c;
    int ctr=0;
    printf("How many rows and columns are in this matrix? ");
    scanf("%d %d", &r, &c);
    printf("Enter the elements in the matrix :\n");
    for(i=0;i<r;i++)
    {
        for(j=0;j<c;j++)
        {
            scanf("%d",&arr[i][j]);
            if (arr[i][j]==0)
                ++ctr;
        }
    }
    if (ctr>((r*c)/2))
        printf ("The given matrix is a sparse matrix. \n");
    else
        printf ("The given matrix is not a sparse matrix.\n");
    printf ("There are %d number of zeros in the matrix.\n",ctr);
}
```

 Now, let's go behind the scenes to understand the code better.

# How it works...

Because we don't want to fix the size of the matrix, we will define a macro called `max` of value 100\. A matrix, or a two-dimensional array called **arr**, is defined of the order max x max. You will be prompted to enter the order of the matrix, for which you can again enter any value up to 100.

Let's assume that you’ve specified the order of the matrix as 4 x 4\. You will be prompted to enter elements in the matrix. The values entered in the matrix will be in row-major order. After entering the elements, the matrix **arr** should look like *F**igure 1.22,* as follows:

![](img/2264c2f1-7355-4d7e-98f3-ae5f178a03f9.png)

Figure 1.22

A counter called `ctr` is created and is initialized to `0`. Using nested loops, each element of matrix `arr` is checked and the value of `ctr` is incremented if any element is found to be 0\. Thereafter, using the `if else` statement, we will check whether the count of zero values is more than non-zero values. If the count of zero values is more than non-zero values, then the message will be displayed on the screen as follows:

```cpp
The given matrix is a sparse matrix
```

However, failing that, the message will be displayed on the screen as follows:

```cpp
The given matrix is not a sparse matrix
```

Let's use GCC to compile the `sparsematrix.c` program as follows:

```cpp
D:\CBook>gcc sparsematrix.c -o sparsematrix
```

Let's run the generated executable file, `sparsematrix.exe`, to see the output of the program:

```cpp
D:\CBook>./sparsematrix
How many rows and columns are in this matrix? 4 4
Enter the elements in the matrix :
0 1 0 0
5 0 0 9
0 0 3 0
2 0 4 0
The given matrix is a sparse matrix.
There are 10 zeros in the matrix.
```

Okay. Let's run the program again to see the output when the count of non-zero values is higher:

```cpp
D:\CBook>./sparsematrix
How many rows and columns are in this matrix? 4 4
Enter the elements in the matrix:
1 0 3 4
0 0 2 9
8 6 5 1
0 7 0 4
The given matrix is not a sparse matrix.
There are 5 zeros in the matrix.
```

Voilà! We've successfully identified a sparse and a non-sparse matrix.

# There's more...

How about finding an identity matrix, in other words, finding out whether the matrix entered by the user is an identity matrix or not. Let me tell you—a matrix is said to be an identity matrix if it is a square matrix and all the elements of the principal diagonal are ones and all other elements are zeros. An identity matrix of the order **3 x 3** may appear as follows:

![](img/16d213c2-8deb-429b-a21c-bfabc3f90743.png)

Figure 1.23

In the preceding diagram, you can see that the principal diagonal elements of the matrix are 1's and the rest of them are 0's. The index or subscript location of the principal diagonal elements will be `arr[0][0]`, `arr[1][1]`, and `arr[2][2]`, so the following procedure is followed to find out whether the matrix is an identity matrix or not:

*   Checks that if the index location of the row and column is the same, in other words, if the row number is 0 and the column number, too, is 0, then at that index location, [0][0], the matrix element must be `1`. Similarly, if the row number is 1 and the column number, too, is 1, that is, at the [1][1] index location, the matrix element must be `1`.
*   Verify that the matrix element is `0` at all the other index locations.

If both the preceding conditions are met, then the matrix is an identity matrix, or else it is not.

The `identitymatrix.c` program to establish whether the entered matrix is an identity matrix or not is given as follows:

```cpp
    #include <stdio.h>
#define max 100
/* All the elements of the principal diagonal of the  Identity matrix  are ones and rest all are zero elements  */
void main ()
{
    static int arr[max][max];
    int i,j,r,c, bool;
    printf("How many rows and columns are in this matrix ? ");
    scanf("%d %d", &r, &c);
    if (r !=c)
    {
        printf("An identity matrix is a square matrix\n");
        printf("Because this matrix is not a square matrix, so it is not an 
           identity matrix\n");
    }
    else
    {
        printf("Enter elements in the matrix :\n");
        for(i=0;i<r;i++)
        {
            for(j=0;j<c;j++)
            {
                scanf("%d",&arr[i][j]);
            }
        }
        printf("\nThe entered matrix is \n");
        for(i=0;i<r;i++)
        {
            for(j=0;j<c;j++)
            {
                printf("%d\t",arr[i][j]);
            }
            printf("\n");
        }
        bool=1;
        for(i=0;i<r;i++)
        {
            for(j=0;j<c;j++)
            {
                if(i==j)
                {
                    if(arr[i][j] !=1)
                    {
                        bool=0;
                        break;
                    }
                }
                else
                {
                    if(arr[i][j] !=0)
                    {
                        bool=0;
                        break;
                    }
                }
            }
        }
        if(bool)
            printf("\nMatrix is an identity matrix\n");                             
        else 
            printf("\nMatrix is not an identity matrix\n");                
    }
}
```

Let's use GCC to compile the `identitymatrix.c` program as follows:

```cpp
D:\CBook>gcc identitymatrix.c -o identitymatrix
```

No error is generated. This means the program is compiled perfectly and an executable file is generated. Let's run the generated executable file. First, we will enter a non-square matrix:

```cpp
D:\CBook>./identitymatrix
How many rows and columns are in this matrix ? 3 4
An identity matrix is a square matrix 
Because this matrix is not a square matrix, so it is not an identity matrix
```

Now, let's run the program again; this time, we will enter a square matrix

```cpp
D:\CBook>./identitymatrix 
How many rows and columns are in this matrix ? 3 3 
Enter elements in the matrix : 
1 0 1 
1 1 0 
0 0 1 

The entered matrix is 
1       0       1 
1       1       0 
0       0       1 

Matrix is not an identity matrix
```

Because a non-diagonal element in the preceding matrix is `1`, it is not an identity matrix. Let's run the program again:

```cpp
D:\CBook>./identitymatrix 
How many rows and columns are in this matrix ? 3 3
Enter elements in the matrix :
1 0 0
0 1 0
0 0 1
The entered matrix is
1       0       0
0       1       0
0       0       1
Matrix is an identity matrix
```

Now, let's move on to the next recipe!

# Merging two sorted arrays into a single array

In this recipe, we will learn to merge two sorted arrays into a single array so that the resulting merged array is also in sorted form.

# How to do it…

1.  Let's assume there are two arrays, **p** and **q**, of a certain length. The length of the two arrays can differ. Both have some sorted elements in them, as shown in *Figure 1.24*:

![](img/b01103bf-76f9-4a4e-8696-87533cd2d6b0.png)

Figure 1.24

2.  The merged array that will be created from the sorted elements of the preceding two arrays will be called array **r**. Three subscripts or index locations will be used to point to the respective elements of the three arrays.
3.  Subscript `i` will be used to point to the index location of array `p`. Subscript `j` will be used to point to the index location of array `q` and subscript `k` will be used to point to the index location of array `r`. In the beginning, all three subscripts will be initialized to `0`.

4.  The following three formulas will be applied to get the merged sorted array:
    1.  The element at `p[i]` is compared with the element at `q[j]`. If `p[i]` is less than `q[j]`, then `p[i]` is assigned to array `r`, and the indices of arrays `p` and `r` are incremented so that the following element of array `p` is picked up for the next comparison as follows:

```cpp
r[k]=p[i];
i++;
k++
```

2.  If `q[j]` is less than `p[i]`, then `q[j]` is assigned to array `r`, and the indices of arrays `q` and `r`  are incremented so that the following element of array `q` is picked up for the next comparison as follows:

```cpp
r[k]=q[j];
i++;
k++
```

3.  If `p[i]` is equal to `q[j]`, then both the elements are assigned to array `r`.  `p[i]` is added to `r[k]`. The values of the `i` and `k` indices are incremented.  `q[j]` is also added to `r[k]`, and the indices of the `q` and `r` arrays are incremented. Refer to the following code snippet:

```cpp
r[k]=p[i];
i++;
k++
r[k]=q[j];
i++;
k++
```

5.  The procedure will be repeated until either of the arrays gets over. If any of the arrays is over, the remainder of the elements of the other array will be simply appended to the array `r`.

The `mergetwosortedarrays.c` program for merging two sorted arrays is as follows:

```cpp
#include<stdio.h>
#define max 100

void main()
{
    int p[max], q[max], r[max];
    int m,n;
    int i,j,k;
    printf("Enter length of first array:");
    scanf("%d",&m);
    printf("Enter %d elements of the first array in sorted order     
    \n",m);
    for(i=0;i<m;i++)
        scanf("%d",&p[i]);
    printf("\nEnter length of second array:");
    scanf("%d",&n);
    printf("Enter %d elements of the second array in sorted 
    order\n",n);
    for(i=0;i<n;i++ )
        scanf("%d",&q[i]);
    i=j=k=0;
    while ((i<m) && (j <n))
    {
        if(p[i] < q[j])
        {
            r[k]=p[i];
            i++;
            k++;
        }
        else
        {
            if(q[j]< p[i])
            {
                r[k]=q[j];
                k++;
                j++;
            }
            else
            {
                r[k]=p[i];
                k++;
                i++;
                r[k]=q[j];
                k++;
                j++;
            }
        }
    }
    while(i<m)
    {
        r[k]=p[i];
        k++;
        i++;
    }
    while(j<n)
    {
        r[k]=q[j];
        k++;
        j++;
    }
    printf("\nThe combined sorted array is:\n");
    for(i = 0;i<k;i++)
        printf("%d\n",r[i]);
}
```

Now, let's go behind the scenes to understand the code better.

# How it works...

A macro called `max` is defined of size `100`. Three arrays, `p`, `q`, and `r`, are defined of size `max`. You will first be asked to enter the size of the first array, `p`, followed by the sorted elements for array `p`. The process is repeated for the second array `q`.

Three indices, `i`, `j` and `k`, are defined and initialized to `0`. The three indices will point to the elements of the three arrays, `p`, `q`, and `r`, respectively.

The first elements of arrays **p** and **q**, in other words, **p[0]**  and **q[0]**, are compared and the smaller one is assigned to array **r**.

Because **q[0]** is smaller than **p[0]**, **q[0]** is added to array **r**, and the indices of arrays **q** and **r** are incremented for the next comparison as follows:

![](img/40d6494f-fdfb-46a2-b174-b2f4b4621945.png)

Figure 1.25

Next, **p[0]** will be compared with **q[1]**. Because **p[0]** is smaller than **q[1]**, the value at **p[0]** will be assigned to array **r** at **r[1]**:

![](img/d2476164-5638-4534-86c9-48ab0659aae0.png)

Figure 1.26

Then, **p[1]** will be compared with **q[1]**. Because **q[1]** is smaller than **p[1]**, **q[1]** will be assigned to array **r**, and the indices of the **q** and **r** arrays will be incremented for the next comparisons (refer to the following diagram):

![](img/0f4ee789-bcae-4267-8aeb-1d6a1a5ab427.png)

Figure 1.27

Let's use GCC to compile the `mergetwosortedarrays.c` program as follows:

```cpp
D:\CBook>gcc mergetwosortedarrays.c -o mergetwosortedarrays
```

Now, let's run the generated executable file, `mergetwosortedarrays.exe`, in order to see the output of the program:

```cpp
D:\CBook>./mergetwosortedarrays
Enter length of first array:4
Enter 4 elements of the first array in sorted order
4
18
56
99

Enter length of second array:5
Enter 5 elements of the second array in sorted order
1
9
80
200
220

The combined sorted array is:
1
4
9
18
56
80
99
200
220
```

Voilà! We've successfully merged two sorted arrays into one.