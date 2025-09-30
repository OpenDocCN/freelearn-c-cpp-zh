# Deep Dive into Pointers

Pointers have been the popular choice among programmers when it comes to using memory in an optimized way. Pointers have made it possible to access the content of any variable, array, or data type. You can use pointers for low-level access to any content and improve the overall performance of an application.

In this chapter, we will look at the following recipes on pointers:

*   Reversing a string using pointers
*   Finding the largest value in an array using pointers
*   Sorting a singly linked list
*   Finding the transpose of a matrix using pointers 
*   Accessing a structure using a pointer

Before we start with the recipes, I would like to discuss a few things related to how pointers work.

# What is a pointer?

A pointer is a variable that contains the memory address of another variable, array, or string. When a pointer contains the address of something, it is said to be pointing at that thing. When a pointer points at something, it receives the right to access the content of that memory address. The question now is—why do we need pointers at all?

We need them because they do the following:

*   Facilitate the dynamic allocation of memory
*   Provide an alternative way to access a data type (apart from variable names, you can access the content of a variable through pointers)
*   Make it possible to return more than one value from a function

For example, consider an `i` integer variable:

```cpp
int i;
```

When you define an integer variable, two bytes will be allocated to it in memory. This set of two bytes can be accessed by a memory address. The value assigned to the variable is stored inside that memory location, as shown in the following diagram:

![](img/3c87d29f-a0d3-402a-bef6-6e993c1cfa4f.png)

Figure 4.1

In the preceding diagram, **1000** represents the memory address of the **i** variable. Though, in reality, memory address is quite big and is in hex format, for the sake of simplicity, I am taking a small integer number, **1000**. The value of **10** is stored inside the memory address, **1000**.

Now, a `j` integer pointer can be defined as follows:

```cpp
int *j;
```

This `j` integer pointer can point to the `i` integer through the following statement:

```cpp
j=&i;
```

The `&` (ampersand) symbol represents the address, and the address of **i** will be assigned to the **j** pointer, as shown in the following diagram. The **2000** address is assumed to be the address of the **j** pointer and the address of the **i** pointer, that is, **1000**, is stored inside the memory location assigned to the **j** pointer, as shown in the following diagram:

![](img/ce63d3dc-5a76-419e-bbda-59f3854d0d25.png)

Figure 4.2

The address of the `i` integer can be displayed by the following statements:

```cpp
printf("Address of i is %d\n", &i); 
printf("Address of i is %d\n", j);
```

To display the contents of `i`, we can use the following statements:

```cpp
printf("Value of i is %d\n", i);
printf("Value of i is %d\n", *j);
```

In the case of pointers, `&` (ampersand) represents the memory address and `*` (asterisk) represents content in the memory address.

We can also define a pointer to an integer pointer by means of the following statement:

```cpp
int **k;
```

This pointer to a `k` integer pointer can point to a `j` integer pointer using the following statement:

```cpp
k=&j;
```

Through the previous statement, the address of the **j** pointer will be assigned to the pointer to a **k** integer pointer, as shown in the following diagram. The value of **3000** is assumed to be the memory address of **k**:

![](img/ccd2200b-6047-49c2-9618-c883acb9e939.png)

Figure 4.3

Now, when you display the value of `k`, it will display the address of `j`:

```cpp
printf("Address of j =%d %d \n",&j,k);
```

To display the address of `i` through `k`, we need to use `*k`, because `*k` means that it will display the contents of the memory address pointed at by `k`. Now, `k` is pointing at `j` and the content in `j` is the address of `i`:

```cpp
printf("Address of i = %d %d %d\n",&i,j,*k);
```

Similarly, to display the value of `i` through `k`, `**k` has to be used as follows:

```cpp
printf("Value of i is %d %d %d %d \n",i,*(&i),*j,**k);
```

Using pointers enables us to access content precisely from desired memory locations. But allocating memory through pointers and not releasing it when the job is done may lead to a problem called **memory leak**. A memory leak is a sort of resource leak. A memory leak can allow unauthorized access of the memory content to hackers and may also block some content from being accessed even though it is present. 

Now, let's begin with the first recipe of this chapter.

# Reversing a string using pointers

In this recipe, we will learn to reverse a string using pointers. The best part is that we will not reverse the string and copy it onto another string, but we will reverse the original string itself.

# How to do it…

1.  Enter a string to assign to the `str` string variable as follows:

```cpp
printf("Enter a string: ");
scanf("%s", str);
```

2.  Set a pointer to point at the string, as demonstrated in the following code. The pointer will point at the memory address of the string's first character:

```cpp
ptr1=str;
```

3.  Find the length of the string by initializing an `n` variable to `1`. Set a `while` loop to execute when the pointer reaches the null character of the string as follows:

```cpp
n=1;
while(*ptr1 !='\0')
{
```

4.  Inside the `while` loop, the following actions will be performed:

*   The pointer is moved one character forward.
*   The value of the `n` variable is incremented by 1:

```cpp
ptr1++;
n++;
```

5.  The pointer will be at the null character, so move the pointer one step back to make it point at the last character of the string as follows:

```cpp
ptr1--;
```

6.  Set another pointer to point at the beginning of the string as follows:

```cpp
ptr2=str;
```

7.  Exchange the characters equal to half the length of the string. To do that, set a `while` loop to execute for `n/2` times, as demonstrated in the following code snippet:

```cpp
m=1;
while(m<=n/2)
```

8.  Within the `while` loop, the first exchange operations take place; that is, the characters pointed at by our pointers are exchanged:

```cpp
temp=*ptr1;
*ptr1=*ptr2;
*ptr2=temp;
```

9.  After the character exchange, set the second pointer to move forward to point at its next character, that is, at the second character of the string, and move the first pointer backward to make it point at the second to last character as follows:

```cpp
ptr1--;
ptr2++;
```

10.  Repeat this procedure for n/2 times, where `n` is the length of the string. When the `while` loop is finished, we will have the reverse form of the original string displayed on the screen:

```cpp
printf("Reverse string is %s", str);
```

The `reversestring.c` program for reversing a string using pointers is as follows:

```cpp
#include <stdio.h>
void main()
{
    char str[255], *ptr1, *ptr2, temp ;
    int n,m;
    printf("Enter a string: ");
    scanf("%s", str);
    ptr1=str;
    n=1;
    while(*ptr1 !='\0')
    {
        ptr1++;
        n++;
    }
    ptr1--;
    ptr2=str;
    m=1;
    while(m<=n/2)
    {
        temp=*ptr1;
        *ptr1=*ptr2;
        *ptr2=temp;
        ptr1--;
        ptr2++;;
        m++;
    }
    printf("Reverse string is %s", str);
}
```

Now, let's go behind the scenes.

# How it works...

We will be prompted to enter a string that will be assigned to the `str` variable. A string is nothing but a character array. Assuming we enter the name `manish`, each character of the name will be assigned to a location in the array one by one (see *Figure 4.4*). We can see that the first character of the string, the letter **m**, is assigned to the **str[0]** location, followed by the second string character being assigned to the **str[1]** location, and so on. The null character, as usual, is at the end of the string, as shown in the following diagram:

![](img/ec0814c0-ab76-46a8-aed8-43dd924144b6.png)

Figure 4.4

To reverse the string, we will seek the help of two pointers: one will be set to point at the first character of the string, and the other at the final character of the string. So, the first **ptr1** pointer is set to point at the first character of the string as follows:

![](img/367bd11b-28d8-4f37-8df0-d2b397196590.png)

Figure 4.5

The exchanging of the characters has to be executed equal to half the length of the string; therefore, the next step will be to find the length of the string. After finding the string's length, the **ptr1** pointer will be set to move to the final character of the string.

In addition, another **ptr2** pointer is set to point at **m**, the first character of the string, as shown in the following diagram:

![](img/54c260fc-5859-469d-8f3b-9eef62aece1f.png)

Figure 4.6

The next step is to interchange the first and last characters of the string that are being pointed at by the **ptr1** and **ptr2 **pointers (see *Figure 4.7 (a)*). After interchanging the characters pointed at by the **ptr1** and **ptr2** pointers, the string will appear as shown in *Figure 4.7 (b)*:

![](img/ceff3a28-90a8-44c9-b2a6-50f2873702a6.png)

Figure 4.7

After interchanging the first and last characters, we will interchange the second and the second to last characters of the string. To do so, the **ptr2** pointer will be moved forward and set to point at the next character in line, and the **ptr1** pointer will be moved backward and set to point at the second to last character.

You can see in the following *Figure** 4.8 (a)* that the **ptr2** and **ptr1** pointers are set to point at the **a** and **s** characters. Once this is done, another interchanging of the characters pointed at by **ptr2** and **ptr1** will take place. The string will appear as follows (*Figure 4.8 (b)*) after the interchanging of the **a** and **s** characters: 

![](img/1e6860bb-db4c-415b-9302-11dd8b731aaf.png)

Figure 4.8

The only task now left in reversing the string is to interchange the third and the third to last character. So, we will repeat the relocation process of the **ptr2** and **ptr1** pointers. Upon interchanging the **n** and **i** characters of the string, the original **str** string will have been reversed, as follows:

![](img/1778b4ce-5a6c-4d42-9409-061c5b9343c5.png)

Figure 4.9

After applying the preceding steps, if we print the **str** string, it will appear in reverse.

Let's use GCC to compile the `reversestring.c` program as follows:

```cpp
D:\CBook>gcc reversestring.c -o reversestring
```

If you get no errors or warnings, that means the `reversestring.c` program has been compiled into an executable file, called `reversestring.exe`. Let's run this executable file as follows:

```cpp
D:\CBook>./reversestring
Enter a string: manish
Reverse string is hsinam
```

Voilà! We've successfully reversed a string using pointers. Now, let's move on to the next recipe!

# Finding the largest value in an array using pointers

In this recipe, all the elements of the array will be scanned using pointers.

# How to do it…

1.  Define a macro by the name `max` with a size of `100` as follows:

```cpp
#define max 100
```

2.  Define a `p` integer array of a `max` size, as demonstrated in the following code:

```cpp
int p[max]
```

3.  Specify the number of elements in the array as follows:

```cpp
printf("How many elements are there? ");
scanf("%d", &n);
```

4.  Enter the elements for the array as follows:

```cpp
for(i=0;i<n;i++)
    scanf("%d",&p[i]);
```

5.  Define two `mx` and `ptr` pointers to point at the first element of the array as follows:

```cpp
mx=p;
ptr=p;
```

6.  The `mx` pointer will always point at the maximum value of the array, whereas the `ptr` pointer will be used for comparing the remainder of the values of the array. If the value pointed to by the `mx` pointer is smaller than the value pointed at by the `ptr` pointer, the `mx` pointer is set to point at the value pointed at by `ptr`. The `ptr` pointer will then move to point at the next array element as follows:

```cpp
if (*mx < *ptr)
    mx=ptr;
```

7.  If the value pointed at by the `mx` pointer is larger than the value pointed to by the `ptr` pointer, the `mx` pointer is undisturbed and is left to keep pointing at the same value and the `ptr` pointer is moved further to point at the next array element for the following comparison:

```cpp
ptr++;
```

8.  This procedure is repeated until all the elements of the array (pointed to by the `ptr` pointer) are compared with the element pointed to by the `mx` pointer. Finally, the `mx` pointer will be left pointing at the maximum value in the array. To display the maximum value of the array, simply display the array element pointed to by the `mx` pointer as follows:

```cpp
printf("Largest value is %d\n", *mx);
```

The `largestinarray.c` program for finding out the largest value in an array using pointers is as follows:

```cpp
#include <stdio.h>
#define max 100
void main()
{
    int p[max], i, n, *ptr, *mx;
    printf("How many elements are there? ");
    scanf("%d", &n);
    printf("Enter %d elements \n", n);
    for(i=0;i<n;i++)
        scanf("%d",&p[i]);
    mx=p;
    ptr=p;
    for(i=1;i<n;i++)
    {
        if (*mx < *ptr)
            mx=ptr;
        ptr++;
    }
    printf("Largest value is %d\n", *mx);
}
```

Now, let's go behind the scenes.

# How it works...

Define an array of a certain size and enter a few elements in it. These will be the values among which we want to find the largest value. After entering a few elements, the array might appear as follows:

![](img/894bb236-7a56-4ac9-a709-727dd595dc6b.png)

Figure 4.10

We will use two pointers for finding the largest value in the array. Let's name the two pointers **mx** and **ptr**, where the **mx** pointer will be used to point at the maximum value of the array, and the **ptr** pointer will be used for comparing the rest of the array elements with the value pointed at by the **mx** pointer. Initially, both the pointers are set to point at the first element of the array, **p[0]**, as shown in the following diagram:

![](img/eededf67-1b47-4ec1-a827-86ab8936a6d2.png)

Figure 4.11

The **ptr** pointer is then moved to point at the next element of the array, **p[1]**. Then, the values pointed at by the **mx** and **ptr** pointers are compared. This process continues until all the elements of the array have been compared as follows:

![](img/65301ebf-ad4f-4bce-849d-5c8188515914.png)

Figure 4.12

Recall that we want the **mx** pointer to keep pointing at the larger value. Since 15 is greater than 3 (see *Figure 4.13*), the position of the **mx** pointer will be left undisturbed, and the **ptr** pointer will be moved to point at the next element, **p[2]**, as follows:

![](img/7fee9ac7-3775-4f60-a577-5ef01c4ccc8b.png)

Figure 4.13

Again, the values pointed at by the **mx** and **ptr** pointers, which are the values 15 and 70 respectively, will be compared. Now, the value pointed at by the **mx** pointer is smaller than the value pointed at by the **ptr** pointer. So, the **mx** pointer will be set to point at the same array element as **ptr** as follows:

![](img/687c6f9c-b6d8-48ef-8d9a-6eaa819c3e9c.png)

Figure 4.14

The comparison of the array elements will continue. The idea is to keep the **mx** pointer pointing at the largest element in the array, as shown in the following diagram:

![](img/5d818dea-224c-45ee-a302-38dc01f39538.png)

Figure 4.15

As shown in *Figure 4.15*, **70** is greater than **20**, so the **mx** pointer will remain at **p[2]**, and the **ptr** pointer will move to the next element, **p[4]**. Now, the **ptr** pointer is pointing at the last array element. So, the program will terminate, displaying the last value pointed at by the **mx** pointer, which also happens to be the largest value in the array.

Let's use GCC to compile the `largestinarray.c` program as the following statement:

```cpp
D:\CBook>gcc largestinarray.c -o largestinarray
```

If you get no errors or warnings, that means that the `largestinarray.c` program has been compiled into an executable file, `largestinarray.exe`. Let's now run this executable file as follows:

```cpp
D:\CBook>./largestinarray
How many elements are there? 5
Enter 5 elements
15
3
70
35
20
Largest value is 70
You can see that the program displays the maximum value in the array
```

Voilà! We've successfully found the largest value in an array using pointers. Now, let's move on to the next recipe!

# Sorting a singly linked list

In this recipe, we will learn how to create a singly linked list comprising integer elements, and then we will learn how to sort this linked list in ascending order. 

A singly linked list consists of several nodes that are connected through pointers. A node of a singly linked list might appear as follows:

![](img/29c0aa98-0d41-496d-adfd-c0ca9c167791.png)

Figure 4.16

As you can see, a node of a singly linked list is a structure composed of two parts:

*   **Data: **This can be one or more variables (also called members) of integer, float, string, or any data type. To keep the program simple, we will take **data** as a single variable of the integer type.
*   **Pointer**: This will point to the structure of the type node. Let's call this pointer **next** in this program, though it can be under any name.

We will use bubble sort for sorting the linked list. Bubble sort is a sequential sorting technique that sorts by comparing adjacent elements. It compares the first element with the second element, the second element with the third element, and so on. The elements are interchanged if they are not in the preferred order. For example, if you are sorting elements into ascending order and the first element is larger than the second element, their values will be interchanged. Similarly, if the second element is larger than the third element, their values will be interchanged too.

This way, you will find that, by the end of the first iteration, the largest value will *bubble* down towards the end of the list. After the second iteration, the second largest value will be *bubbled* down to the end of the list. In all, n-1 iterations will be required to sort the n elements using bubble sort algorithm.

Let's understand the steps in creating and sorting a singly linked list.

# How to do it…

1.  Define a node comprising two members—`data` and `next`. The data member is for storing integer values and the next member is a pointer to link the nodes as follows:

```cpp
struct node
{
  int data;
  struct node *next;
};
```

2.  Specify the number of elements in the linked list. The value entered will be assigned to the `n` variable as follows:

```cpp
printf("How many elements are there in the linked list ?");
scanf("%d",&n);
```

3.  Execute a `for` loop for `n` number of times. Within the `for` loop, a node is created by the name `newNode`. When asked, enter an integer value to be assigned to the data member of `newNode` as follows:

```cpp
newNode=(struct node *)malloc(sizeof(struct node));
scanf("%d",&newNode->data);
```

4.  Two pointers, `startList` and `temp1`, are set to point at the first node. The `startList` pointer will keep pointing at the first node of the linked list. The `temp1` pointer will be used to link the nodes as follows:

```cpp
startList = newNode;
temp1=startList;
```

5.  To connect the newly created nodes, the following two tasks are performed:

*   The next member of `temp1` is set to point at the newly created node.
*   The `temp1` pointer is shifted to point at the newly created node as follows:

```cpp
temp1->next = newNode;
temp1=newNode;
```

6.  When the `for` loop gets over, we will have a singly linked list with its first node pointed at by `startList`, and the next pointer of the last node pointing at NULL. This linked list is ready to undergo the sorting procedure. Set a `for` loop to execute from `0` until `n-2` that is equal to n-1 iterations as follows:

```cpp
for(i=n-2;i>=0;i--)
```

7.  Within the `for` loop, to compare values, use two pointers, `temp1` and `temp2`. Initially, `temp1` and `temp2` will be set to point at the first two nodes of the linked list, as shown in the following code snippet:

```cpp
temp1=startList;
temp2=temp1->next;
```

8.  Compare the nodes pointed at by `temp1` and `temp2` in the following code:

```cpp
if(temp1->data > temp2->data)
```

9.  After comparing the first two nodes, the `temp1` and `temp2` pointers will be set to point at the second and third nodes, and so on:

```cpp
temp1=temp2;
temp2=temp2->next;
```

10.  The linked list has to be arranged in ascending order, so the data member of `temp1` must be smaller than the data member of `temp2`. In case the data member of `temp1` is larger than the data member of `temp2`, the interchanging of the values of the data members will be done with the help of a temporary variable, `k`, as follows:

```cpp
k=temp1->data;
temp1->data=temp2->data;
temp2->data=k;
```

11.  After n-1 performing iterations of comparing and interchanging consecutive values, if the first value in the pair is larger than the second, all the nodes in the linked list will be arranged in ascending order. To traverse the linked list and to display the values in ascending order, a temporary `t` pointer is set to point at the node pointed at by `startList`, that is, at the first node of the linked list, as follows:

```cpp
t=startList;
```

12.  A `while` loop is executed until the `t` pointer reaches `NULL`. Recall that the next pointer of the last node is set to NULL, so the `while` loop will execute until all the nodes of the linked list are traversed as follows:

```cpp
while(t!=NULL)
```

13.  Within the `while` loop, the following two tasks will be performed:

*   The data member of the node pointed to by the `t` pointer is displayed.
*   The `t` pointer is moved further to point at its next node:

```cpp
printf("%d\t",t->data);
t=t->next;
```

The `sortlinkedlist.c` program for creating a singly linked list, followed by sorting it in ascending order, is as follows:

```cpp
/* Sort the linked list by bubble sort */
#include<stdio.h>
#include <stdlib.h>
struct node
{
  int data;
  struct node *next;
};
void main()
{
    struct node *temp1,*temp2, *t,*newNode, *startList;
    int n,k,i,j;
    startList=NULL;
    printf("How many elements are there in the linked list ?");
    scanf("%d",&n);
    printf("Enter elements in the linked list\n");
    for(i=1;i<=n;i++)
    {
        if(startList==NULL)
        {
            newNode=(struct node *)malloc(sizeof(struct node));
            scanf("%d",&newNode->data);
            newNode->next=NULL;
            startList = newNode;
            temp1=startList;
        }
        else
        {
            newNode=(struct node *)malloc(sizeof(struct node));
            scanf("%d",&newNode->data);
            newNode->next=NULL;
            temp1->next = newNode;
            temp1=newNode;
        }
    }
    for(i=n-2;i>=0;i--)
    {
        temp1=startList;
        temp2=temp1->next;
        for(j=0;j<=i;j++)
        {
            if(temp1->data > temp2->data)
            {
                k=temp1->data;
                temp1->data=temp2->data;
                temp2->data=k;
            }
            temp1=temp2;
            temp2=temp2->next;
        }
    }
    printf("Sorted order is: \n");
    t=startList;
    while(t!=NULL)
    {
        printf("%d\t",t->data);
        t=t->next;
    }
}
```

Now, let's go behind the scenes.

# How it works...

This program is performed in two parts—the first part is the creation of a singly linked list, and the second part is the sorting of the linked list.

Let's start with the first part.

# Creating a singly linked list

We will start by creating a new node by the name of **newNode**. When prompted, we will enter the value for its data member and then set the next **newNode** pointer to **NULL** ( as shown in *Figure 4.17*). This next pointer will be used for connecting with other nodes (as we will see shortly):

![](img/5a78052d-7c8b-420f-89bd-e343ff6634af.png)

Figure 4.17

After the first node is created, we will make the following two pointers point at it as follows:

*   **startList**: To traverse the singly linked list, we will need a pointer that points at the first node of the list. So, we will define a pointer called **startList** and set it to point at the first node of the list.
*   **temp1**: In order to connect with the next node, we will need one more pointer. We will call this pointer **temp1**, and set it to point at the **newNode** (see *Figure 4.18*):

![](img/79419819-e2e7-4be7-a8af-b55050370191.png)

Figure 4.18

We will now create another node for the linked list and call that **newNode** as well. The pointer can point to only one structure at a time. So, the moment we create a new node, the **newNode** pointer that was pointing at the first node will now point at the recently created node. We will be prompted to enter a value for the data member of the new node, and its next pointer will be set to **NULL**.

You can see in the following diagram that the two pointers, **startList** and **temp1**, are pointing at the first node and the **newNode** pointer is pointing at the newly created node. As stated earlier, **startList** will be used for traversing the linked list and **temp1** will be used for connecting with the newly created node as follows:

![](img/bdd0494c-d8ba-414a-968d-933f5970834a.png)

Figure 4.19

To connect the first node with **newNode**, the next pointer of **temp1** will be set to point at **newNode** (see *Figure 4.20 (a)*). After connecting with **newNode**, the **temp**`1` pointer will be moved further and set to point at **newNode** (see *Figure 4.20 (b)*) so that it can be used again for connecting with any new nodes that may be added to the linked list in future:

![](img/b1a8d265-9019-4157-a2c4-5f33dc5eb079.png)

Figure 4.20

Steps three and four will be repeated for the rest of the nodes of the linked list. Finally, the singly linked list will be ready and will look something like this:

![](img/50816494-7dcc-42b4-ab85-58d6a357caac.png)

Figure 4.21

Now that we have created the singly linked list, the next step is to sort the linked list in ascending order.

# Sorting the singly linked list

We will use the bubble sort algorithm for sorting the linked list. In the bubble sort technique, the first value is compared with the second value, the second is compared with the third value, and so on. If we want to sort our list in ascending order, then we will need to keep the smaller values toward the top when comparing the values.

Therefore, while comparing the first and second values, if the first value is larger than the second value, then their places will be interchanged. If the first value is smaller than the second value, then no interchanging will happen, and the second and third values will be picked up for comparison.

There will be n-1 iterations of such comparisons, meaning if there are five values, then there will be four iterations of such comparisons; and after every iteration, the last value will be left out—that is, it will not be compared as it reaches its destination. The destination here means the location where the value must be kept when arranged in ascending order.

# The first iteration

To sort the linked list, we will employ the services of two pointers—**temp1** and **temp2**. The **temp1** pointer is set to point at the first node, and **temp2** is set to point at the next node as follows:

![](img/49b5a9f5-01e6-4ab5-a0cf-57fb514b5fb9.png)

Figure 4.22

We will be sorting the linked list in ascending order, so we will keep the smaller values toward the beginning of the list. The data members of **temp1** and **temp2** will be compared. Because `temp1->data` is greater than `temp2->data`, that is, the data member of **temp1** is larger than the data member of **temp2**, their places will be interchanged (see the following diagram). After interchanging the data members of the nodes pointed at by **temp1** and **temp2**, the linked list will appear as follows:

![](img/3cff457d-883c-4c46-84f1-c1d1d23f9449.png)

Figure 4.23

After this, the two pointers will shift further, that is, the **temp1** pointer will be set to point at **temp2**, and the **temp2** pointer will be set to point at its next node. We can see in *Figure 4.24 (a)* that the **temp1** and **temp2** pointers are pointing at the nodes with the values 3 and 7, respectively. We can also see that `temp1->data` is less than `temp2->data`, that is, 3 < 7\. Since the data member of **temp1** is already smaller than the data member of **temp2**, no interchanging of values will take place and the two pointers will simply move one step further (see *Figure 4.24 (b)*).

Now, because 7 > 4, their places will be interchanged. The values of data members pointed at by **temp1** and **temp2** will interchange as follows (*Figure 4.24 (c)*):

![](img/69a89064-f366-4c09-9772-cab43be27292.png)

Figure 4.24

After that, the **temp1** and **temp2** pointer will be shifted one step further, that is, **temp1** will point at **temp2**, and **temp2** will move onto its next node. We can see in the following *Figure 4.25 (a)* that **temp1** and **temp2** are pointing at the nodes with the values 7 and 2, respectively. Again, the data members of **temp1** and **temp2** will be compared. Because `temp1->data` is greater than `temp2->data`, their places will be interchanged. *Figure 4.25 (b)* shows the linked list after interchanging values of the data members:

![](img/a562a203-d4dc-4180-b967-ee74ee062719.png)

Figure 4.25

This was the first iteration, and you can notice that after this iteration, the largest value, 7, has been set to our desired location—at the end of the linked list. This also means that in the second iteration, we will not have to compare the last node. Similarly, after the second iteration, the second highest value will reach or is set to its actual location. The second highest value in the linked list is 4, so after the second iteration, the four node will just reach the seven node. How? Let's look at the second iteration of bubble sort.

# The second iteration

We will begin the comparison by comparing first two nodes, so the **temp1** and **temp2** pointers will be set to point at the first and second nodes of the linked list, respectively (see *Figure 4.26 (a)*). The data members of **temp1** and **temp2** will be compared. As is clear, `temp1->data` is less than `temp2->data` (that is, 1 < 7), so their places will not be interchanged. Thereafter, the **temp1** and **temp2** pointers will shift one step further. We can see in *Figure 4.26 (b)* that the **temp1** and **temp2** pointers are set to point at nodes of the values 3 and 4, respectively:

![](img/4ff89217-6f3f-49af-acd3-90db85b3ac72.png)

Figure 4.26

Once again, the data members of the **temp1** and **temp2** pointers will be compared. Because `temp1->data` is less than `temp2->data`, that is, 3 < 4 , their places will again not be interchanged and the **temp1** and **temp2** pointers will, again, shift one step further. That is, the **temp1** pointer will be set to point at **temp2**, and **temp2** will be set to point at its next node. You can see in *Figure 4.27 (a)* that the **temp1** and **temp2** pointers are set to point at nodes with the values 4 and 2, respectively. Because 4 > 2, their places will be interchanged. After interchanging the place of these values, the linked list will appear as follows in *Figure 4.27 (b)*:

![](img/d8a88cfd-1f57-4b31-9669-616cc3bf8175.png)

Figure 4.27

This is the end of the second iteration, and we can see that the second largest value, four, is set to our desired location as per ascending order. So, with every iteration, one value is being set at the required location. Accordingly, the next iteration will require one comparison less.

# The third and fourth iterations

In the third iteration, we will only need to do the following comparisons:

1.  Compare the first and second nodes
2.  Compare the second and third nodes

After the third iteration, the third largest value, that is, three, will be set at our desired location, that is, just before node four.

In the fourth, and final, iteration, only the first and second nodes will be compared. The linked list will be sorted in ascending order as follows after the fourth iteration:

![](img/b85ae8be-01bc-46d5-b8e6-0ec16f1bd232.png)

Figure 4.28

Let's use GCC to compile the `sortlinkedlist.c` program as follows:

```cpp
D:\CBook>gcc sortlinkedlist.c -o sortlinkedlist
```

If you get no errors or warnings, that means that the `sortlinkedlist.c` program has been compiled into an executable file, `sortlinkedlist.exe`. Let's run this executable file as follows:

```cpp
D:\CBook>./sortlinkedlist
How many elements are there in the linked list ?5
Enter elements in the linked list
3
1
7
4
2
Sorted order is:
1       2       3       4       7
```

Voilà! We've successfully created and sorted a singly linked list. Now, let's move on to the next recipe!

# Finding the transpose of a matrix using pointers

The best part of this recipe is that we will not only display the transpose of the matrix using pointers, but we will also create the matrix itself using pointers. 

The transpose of a matrix is a new matrix that has rows equal to the number of columns of the original matrix and columns equal to the number of rows. The following diagram shows you a matrix of order **2 x 3** and its transpose, of order **3 x 2**:

![](img/841d4994-ab2c-4675-b605-c4b0a36f54f9.png)

Figure 4.29

Basically, we can say that, upon converting the rows into columns and columns into rows of a matrix, you get its transpose.

# How to do it…

1.  Define a matrix of 10 rows and 10 columns as follows (you can have a bigger matrix if you wish):

```cpp
int a[10][10]
```

2.  Enter the size of the rows and columns as follows:

```cpp
    printf("Enter rows and columns of matrix: ");
    scanf("%d %d", &r, &c);
```

3.  Allocate memory locations equal to `r *c` quantity for keeping the matrix elements as follows:

```cpp
    ptr = (int *)malloc(r * c * sizeof(int));
```

4.  Enter elements of the matrix that will be assigned sequentially to each allocated memory as follows:

```cpp
    for(i=0; i<r; ++i)
    {
        for(j=0; j<c; ++j)
        {
            scanf("%d", &m);
             *(ptr+ i*c + j)=m;
        }
    }
```

5.  In order to access this matrix via a pointer, set a `ptr` pointer to point at the first memory location of the allocated memory block, as shown in *Figure 4.30*. The moment that the `ptr` pointer is set to point at the first memory location, it will automatically get the address of the first memory location, so `1000` will be assigned to the `ptr` pointer:

![](img/95534adc-2f68-4dea-87e2-f55e992fccd0.png)

Figure 4.30

6.  To access these memory locations and display their content, use the `*(ptr +i*c + j)` formula within the nested loop, as shown in this code snippet:

```cpp
for(i=0; i<r; ++i)
{
    for(j=0; j<c; ++j)
    {
        printf("%d\t",*(ptr +i*c + j));
    }
    printf("\n");
}
```

7.  The value of the `r` row is assumed to be two, and that of column `c` is assumed to be three. With values of `i=0` and `j=0`, the formula will compute as follows:

```cpp
*(ptr +i*c + j);
*(1000+0*3+0)
*1000
```

It will display the content of the memory address, `1000`.

When the value of `i=0` and `j=1`, the formula will compute as follows:

```cpp
*(ptr +i*c + j);
*(1000+0*3+1)
*(1000+1)
*(1002)
```

We will first get `*(1000+1)`, because the `ptr` pointer is an integer pointer, and it will jump two bytes every time we add the value `1` to it at every memory location, from which we will get `*(1002)`, and it will display the content of the memory location `1002`.

Similarly, the value of `i=0` and `j=2`will lead to `*(1004)`; that is, the content of the memory location `1004` will be displayed. Using this formula, the value of `i=1` and `j=0` will lead to `*(1006)`; the value of `i=1` and `j=1` will lead to `*(1008)`; and the value of `i=1` and `j=2` will lead to `*(1010)`. So, when the aforementioned formula is applied within the nested loops, the original matrix will be displayed as follows:

![](img/ee37f84f-dad1-41a6-bb97-3af49fe51c91.png)

Figure 4.31

8.  To display the transpose of a matrix, apply the following formula within the nested loops:

```cpp
*(ptr +j*c + i))
```

Again, assuming the values of row (r=2) and column (c=3), the following content of memory locations will be displayed:

| **i** | **j** | **Memory address** |
| 0 | 0 | `1000` |
| 0 | 1 | `1006` |
| 1 | 0 | `1002` |
| 1 | 1 | `1008` |
| 2 | 0 | `1004` |
| 2 | 1 | `1010` |

So, upon applying the preceding formula, the content of the following memory address will be displayed as the following in *Figure 4.32*. And the content of these memory addresses will comprise the transpose of the matrix:

![](img/5c7eb621-d263-4da1-8297-2f06aedaf2b4.png)

Figure 4.32

Let's see how this formula is applied in a program.

The `transposemat.c` program for displaying the transpose of a matrix using pointers is as follows:

```cpp
#include <stdio.h>
#include <stdlib.h>
void main()
{
    int a[10][10],  r, c, i, j, *ptr,m;
    printf("Enter rows and columns of matrix: ");
    scanf("%d %d", &r, &c);
    ptr = (int *)malloc(r * c * sizeof(int));
    printf("\nEnter elements of matrix:\n");
    for(i=0; i<r; ++i)
    {
        for(j=0; j<c; ++j)
        {
            scanf("%d", &m);
             *(ptr+ i*c + j)=m;
        }
    }
    printf("\nMatrix using pointer is: \n");
    for(i=0; i<r; ++i)
    {
        for(j=0; j<c; ++j)
        {
           printf("%d\t",*(ptr +i*c + j));
        }
        printf("\n");
    }
    printf("\nTranspose of Matrix:\n");
    for(i=0; i<c; ++i)
    {
        for(j=0; j<r; ++j)
        {
             printf("%d\t",*(ptr +j*c + i));
        }
        printf("\n");
   }
}
```

Now, let's go behind the scenes.

# How it works...

Whenever an array is defined, the memory allocated to it internally is a sequential memory. Now let's define a matrix of size 2 x 3, as shown in the following diagram. In that case, the matrix will be assigned six consecutive memory locations of two bytes each (see *Figure 4.33*). Why two bytes each? This is because an integer takes two bytes. This also means that if we define a matrix of the float type that takes four bytes, each allocated memory location would consist of four bytes:

![](img/6cb5f6a4-9c19-4b07-b683-5a059474dcd0.png)

Figure 4.33

In reality, the memory address is long and is in hex format; but for simplicity, we will take the memory addresses of integer type and take easy-to-remember numbers, such as **1000**, as memory addresses. After memory address **1000**, the next memory address is **1002** (because an integer takes two bytes).

Now, to display the original matrix elements in row-major form using a pointer, we will need to display the elements of memory locations, **1000**, **1002**, **1004**, and so on:

![](img/83a850f2-c3ec-450c-91b2-32e245017600.png)

Figure 4.34

Similarly, in order to display the transpose of the matrix using a pointer, we will need to display the elements of memory locations; **1000**, **1006**, **1002**, **1008**, **1004**, and **1010**:

![](img/75905466-1375-4fdf-95c2-8857d2998b1d.png)

Figure 4.35

Let's use GCC to compile the `transposemat.c` program as follows:

```cpp
D:\CBook>gcc transposemat.c -o transposemat
```

If you get no errors or warnings, that means that the `transposemat.c` program has been compiled into an executable file, `transposemat.exe`. Let's run this executable file with the following code snippet:

```cpp
D:\CBook>./transposemat
Enter rows and columns of matrix: 2 3

Enter elements of matrix:
1
2
3
4
5
6

Matrix using pointer is:
1       2       3
4       5       6

Transpose of Matrix:
1       4
2       5
3       6
```

Voilà! We've successfully found the transpose of a matrix using pointers. Now, let's move on to the next recipe!

# Accessing a structure using a pointer

In this recipe, we will make a structure that stores the information of an order placed by a specific customer. A structure is a user-defined data type that can store several members of different data types within it. The structure will have members for storing the order number, email address, and password of the customer:

```cpp
struct cart
{  
    int orderno;
    char emailaddress[30];
    char password[30];
};
```

The preceding structure is named `cart`, and comprises three members – `orderno` of the `int` type for storing the order number of the order placed by the customer, and `emailaddress` and `password` of the string type for storing the email address and password of the customer, respectively. Let's begin!

# How to do it…

1.  Define a `cart` structure by the name `mycart`. Also, define two pointers to structure of the `cart` structure, `ptrcart` and `ptrcust`, as shown in the following code snippet:

```cpp
struct cart mycart;
struct cart *ptrcart, *ptrcust;
```

2.  Enter the order number, email address, and password of the customer, and these values will be accepted using the `mycart` structure variable. As mentioned previously, the dot operator (`.`) will be used for accessing the structure members, `orderno`, `emailaddress`, and `password`, through a structure variable as follows:

```cpp
printf("Enter order number: ");
scanf("%d",&mycart.orderno);
printf("Enter email address: ");
scanf("%s",mycart.emailaddress);
printf("Enter password: ");
scanf("%s",mycart.password);
```

3.  Set the pointer to the `ptrcart` structure to point at the `mycart` structure using the `ptrcart=&mycart` statement. Consequently, the pointer to the `ptrcart` structure will be able to access the members of the `mycart` structure by using the arrow (`->`) operator. By using `ptrcart->orderno`, `ptrcart->emailaddress`, and `ptrcart->password`, the values assigned to the `orderno`, `emailaddress`, and `password` structure members are accessed and displayed:

```cpp
printf("\nDetails of the customer are as follows:\n");
printf("Order number : %d\n", ptrcart->orderno);
printf("Email address : %s\n", ptrcart->emailaddress);
printf("Password : %s\n", ptrcart->password);
```

4.  We will also modify the email address and password of the customer by asking them to enter a new email address and password and accept the new details via the pointer to the `ptrcart` structure as follows. Because `ptrcart` is pointing to the `mycart` structure, the new email address and password will overwrite the existing values that were assigned to the structure members of `mycart`:

```cpp
printf("\nEnter new email address: ");
scanf("%s",ptrcart->emailaddress);
printf("Enter new password: ");
scanf("%s",ptrcart->password);
/*The new modified values of orderno, emailaddress and password members are displayed using structure variable, mycart using dot operator (.).*/
printf("\nModified customer's information is:\n");
printf("Order number: %d\n", mycart.orderno);
printf("Email address: %s\n", mycart.emailaddress);
printf("Password: %s\n", mycart.password);
```

5.  Then, define a pointer to the `*ptrcust` structure. Using the following `malloc` function, allocate memory for it. The `sizeof` function will find out the number of bytes consumed by each of the structure members and return the total number of bytes consumed by the structure as a whole:

```cpp
ptrcust=(struct cart *)malloc(sizeof(struct cart));
```

6.  Enter the order number, email address, and password of the customer, and all the values will be assigned to the respective structure members using a pointer to a structure as follows. Obviously, the arrow operator (`->`) will be used for accessing the structure members through a pointer to a structure:

```cpp
printf("Enter order number: ");
scanf("%d",&ptrcust->orderno);
printf("Enter email address: ");
scanf("%s",ptrcust->emailaddress);
printf("Enter password: ");
scanf("%s",ptrcust->password);
```

7.  The values entered by the user are then displayed through the pointer to the `ptrcust` structure again as follows:

```cpp
printf("\nDetails of the second customer are as follows:\n");
printf("Order number : %d\n", ptrcust->orderno);
printf("Email address : %s\n", ptrcust->emailaddress);
printf("Password : %s\n", ptrcust->password);
```

The following `pointertostruct.c` program explains how to access a structure by using a pointer:

```cpp
#include <stdio.h>
#include <stdlib.h>

struct cart
{
    int orderno;
    char emailaddress[30];
    char password[30];
};

void main()
{
    struct cart mycart;
    struct cart *ptrcart, *ptrcust;
    ptrcart = &mycart;
    printf("Enter order number: ");
    scanf("%d",&mycart.orderno);
    printf("Enter email address: ");
    scanf("%s",mycart.emailaddress);
    printf("Enter password: ");
    scanf("%s",mycart.password);
    printf("\nDetails of the customer are as follows:\n");
    printf("Order number : %d\n", ptrcart->orderno);
    printf("Email address : %s\n", ptrcart->emailaddress);
    printf("Password : %s\n", ptrcart->password);

    printf("\nEnter new email address: ");
    scanf("%s",ptrcart->emailaddress);
    printf("Enter new password: ");
    scanf("%s",ptrcart->password);
    printf("\nModified customer's information is:\n");
    printf("Order number: %d\n", mycart.orderno);
    printf("Email address: %s\n", mycart.emailaddress);
    printf("Password: %s\n", mycart.password);

    ptrcust=(struct cart *)malloc(sizeof(struct cart));
    printf("\nEnter information of another customer:\n");
    printf("Enter order number: ");
    scanf("%d",&ptrcust->orderno);
    printf("Enter email address: ");
    scanf("%s",ptrcust->emailaddress);
    printf("Enter password: ");
    scanf("%s",ptrcust->password);
    printf("\nDetails of the second customer are as follows:\n");
    printf("Order number : %d\n", ptrcust->orderno);
    printf("Email address : %s\n", ptrcust->emailaddress);
    printf("Password : %s\n", ptrcust->password);
}
```

Now, let's go behind the scenes.

# How it works...

When you define a variable of the type structure, that variable can access members of the structure in the following format:

```cpp
structurevariable.structuremember
```

You can see a period (`.`) between the structure variable and the structure member. This period (`.`) is also known as a dot operator, or member access operator. The following example will make it clearer:

```cpp
struct cart mycart;
mycart.orderno
```

In the preceding code, you can see that `mycart` is defined as a structure variable of the `cart` structure. Now, the `mycart` structure variable can access the `orderno` member by making the member access operator (`.`).

You can also define a pointer to a structure. The following statement defines `ptrcart` as a pointer to the `cart` structure.

```cpp
struct cart *ptrcart;
```

When the pointer to a structure points to a structure variable, it can access the structure members of the structure variable. In the following statement, the pointer to the `ptrcart` structure points at the address of the `mycart` structure variable:

```cpp
ptrcart = &mycart;
```

Now, `ptrcart` can access the structure members, but instead of the dot operator (`.`), the arrow operator (`->`) will be used. The following statement accesses the `orderno` member of the structure using the pointer to a structure:

```cpp
ptrcart->orderno
```

If you don’t want a pointer to a structure to point at the structure variable, then memory needs to be allocated for a pointer to a structure to access structure members. The following statement defines a pointer to a structure by allocating memory for it:

```cpp
ptrcust=(struct cart *)malloc(sizeof(struct cart));
```

The preceding code allocates memory equal to the size of a `cart` structure, typecasts that memory to be used by a pointer to a `cart` structure, and assigns that allocated memory to `ptrcust`. In other words, `ptrcust` is defined as a pointer to a structure, and it does not need to point to any structure variable, but can directly access the structure members.

Let's use GCC to compile the `pointertostruct.c` program as follows:

```cpp
D:\CBook>gcc pointertostruct.c -o pointertostruct
```

If you get no errors or warnings, that means that the `pointertostruct.c` program has been compiled into an executable file, `pointertostruct.exe`. Let's run this executable file as follows:

```cpp
D:\CBook>./pointertostruct
Enter order number: 1001
Enter email address: bmharwani@yahoo.com
Enter password: gold

Details of the customer are as follows:
Order number : 1001
Email address : bmharwani@yahoo.com
Password : gold

Enter new email address: harwanibm@gmail.com
Enter new password: diamond

Modified customer's information is:
Order number: 1001
Email address: harwanibm@gmail.com
Password: diamond

Enter information of another customer:
Enter order number: 1002
Enter email address: bintu@yahoo.com
Enter password: platinum

Details of the second customer are as follows:
Order number : 1002
Email address : bintu@yahoo.com
Password : platinum
```

Voilà! We've successfully accessed a structure using a pointer.