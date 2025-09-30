# File Handling

Data storage is a mandatory feature in all applications. When we enter any data while running a program, that data is stored as RAM, which means that it is temporary in nature. We will not get that data back when running the program the next time. But what if we want the data to stay there so we can refer to it again when we need it? In this case, we have to store the data.

Basically, we want our data to be stored and to be accessible and available for reuse whenever required. In C, data storage can be done through traditional file handling techniques and through the database system. The following are the two types of file handling available in C:

*   **Sequential file handling**:Data is written in a simple text format and can be read and written sequentially. To read the *n*th line, we have to first read *n*-1 lines.
*   **Random file handling**:Data is written as bytes and can be read or written randomly. We can read or write any line randomly by positioning the file pointer at the desired location.

In this chapter, we will go through the following recipes using file handling:

*   Reading a text file and converting all characters after a period into uppercase
*   Displaying the contents of a random file in reverse order
*   Counting the number of vowels in a file
*   Replacing a word in a file with another word
*   Encrypting a file

Before we start with the recipes, let's review some of the functions we will be using to create our recipes.

# Functions used in file handling

I've divided this section into two parts. In the first part, we will look at the functions specific to the sequential file handling method. In the second, we will look at the functions used for random files.

# Functions commonly used in sequential file handling

The following are some of the functions that are used to open, close, read, and write in a sequential file.

# fopen()

The `fopen()` function is used for opening a file for reading, writing, and doing other operations. Here is its syntax:

```cpp
FILE *fopen (const char *file_name, const char *mode)
```

Here, `file_name` represents the file that we want to work on and `mode` states the purpose for which we want to open the file. It can be any of the following:

*   `r`**:** This opens the file in read mode and sets the file pointer at the first character of the file.
*   `w`**:** This opens the file in write mode. If the file exists, it will be overwritten.
*   `a`**:** Opens the file in append mode. Newly entered data will be added at the end of the file.
*   `r+`**:** This opens the file in read and write mode. The file pointer is set to point at the beginning of the file. The file content will not be deleted if it already exists. It will not create a file if it does not already exist.
*   `w+`**:** This also opens the file in read and write mode. The file pointer is set to point at the beginning of the file. The file content will be deleted if it already exists, but the file will be created if it does not already exist.
*   `a+`**:** This opens a file for reading as well as for appending new content.

The `fopen` function returns a file descriptor that points to the file for performing different operations.

# fclose()

The `fclose()` function is used for closing the file. Here is its syntax:

```cpp
int fclose(FILE *file_pointer)
```

Here, `file_pointer` represents the file pointer that is pointing at the open file.

The function returns a `0` value if the file is successfully closed.

# fgets()

The `fgets()` function is used for reading a string from the specified file. Here is its syntax:

```cpp
char *fgets(char *string, int length, FILE *file_pointer)
```

This function has the following features:

*   `string`**:** This represents the character array to which the data that is read from the file will be assigned.
*   `length`**:** This represents the maximum number of characters that can be read from the file. The *length-1* number of characters will be read from the file. The reading of data from the file will stop either at *length-1* location or at the new line character, `\n`, whichever comes first.
*   `file_pointer`**:** This represents the file pointer that is pointing at the file.

# fputs()

The `fputs()` function is used for writing into the file. Here is its syntax:

```cpp
int fputs (const char *string, FILE *file_pointer)
```

Here, `string` represents the character array containing the data to be written into the file. The `file_pointer` phrase represents the file pointer that is pointing at the file.

# Functions commonly used in random files 

The following functions are used to set the file pointer at a specified location in the random file, indicate the location where the file pointer is pointing currently, and rewind the file pointer to the beginning of the random file.

# fseek()

The `fseek()` function is used for setting the file pointer at the specific position in the file. Here is its syntax:

```cpp
fseek(FILE *file_pointer, long int offset, int location);
```

This function has the following features:

*   `file_pointer`**:** This represents the file pointer that points at the file.
*   `offset`**:** This represents the number of bytes that the file pointer needs to be moved from the position specified by the location parameter. If the value of `offset` is positive, the file pointer will move forward in the file, and if it is negative, the file pointer will move backward from the given position.
*   `location`**:** This is the value that defines the position from which the file pointer needs to be moved. That is, the file pointer will be moved equal to the number of bytes specified by the `offset` parameter from the position specified by the `location` parameter. Its value can be `0`, `1`, or `2`, as shown in the following table:

| **Value** | **Meaning** |
| `0` | The file pointer will be moved from the beginning of the file |
| `1` | The file pointer will be moved from the current position |
| `2` | The file pointer will be moved from the end of the file |

Let's look at the following example. Here, the file pointer will be moved `5` bytes forward from the beginning of the file:

```cpp
fseek(fp,5L,0)
```

In the following example, the file pointer will be moved `5` bytes backward from the end of the file:

```cpp
fseek(fp,-5L,2)
```

# ftell()

The `ftell()` function returns the byte location where `file_pointer` is currently pointing at the file. Here is its syntax:

```cpp
long int ftell(FILE *file_pointer)
```

Here, `file_pointer` is a file pointer pointing at the file.

# rewind()

The `rewind()` function is used for moving the file pointer back to the beginning of the specified file. Here is its syntax:

```cpp
void rewind(FILE *file_pointer)
```

Here, `file_pointer` is a file pointer pointing at the file.

In this chapter, we will learn to use both types of file handling using recipes that make real-time applications.

# Reading a text file and converting all characters after the period into uppercase

Say we have a file that contains some text. We think that there is an anomaly in the text—every first character after the period is in lowercase when it should be in uppercase. In this recipe, we will read that text file and convert each character after the period (`.`) that is, in lowercase into uppercase.

In this recipe, I assume that you know how to create a text file and how to read a text file. If you don't know how to perform these actions, you will find programs for both of them in *Appendix A*.

# How to do it…

1.  Open the sequential file in read-only mode using the following code:

```cpp
    fp = fopen (argv [1],"r");
```

2.  If the file does not exist or does not have enough permissions, an error message will be displayed and the program will terminate. Set this up using the following code:

```cpp
if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
 }
```

3.  One line is read from the file, as shown in the following code:

```cpp
fgets(buffer, BUFFSIZE, fp);
```

4.  Each character of the line is accessed and checked for the presence of periods, as shown in the following code:

```cpp
for(i=0;i<n;i++)
    if(buffer[i]=='.')

```

5.  If a period is found, the character following the period is checked to confirm whether it is in uppercase, as shown in the following code:

```cpp
if(buffer[i] >=97 && buffer[i] <=122)
```

6.  If the character following the period is in lowercase, a value of `32` is subtracted from the ASCII value of the lowercase character to convert it into uppercase, as shown in the following code:

```cpp
buffer[i]=buffer[i]-32;
```

7.  If the line is not yet over, then the sequence from step 4 onward is repeated till step 6; otherwise, the updated line is displayed on the screen, as shown in the following code:

```cpp
puts(buffer);
```

8.  Check whether the end of file has been reached using the following code. If the file is not over, repeat the sequence from step 3:

```cpp
while(!feof(fp))
```

The preceding steps are pictorially explained in the following diagram (*Figure 5.1*):

![](img/192e296b-c627-426f-bf38-7337dbcb9692.png)

Figure 5.1

The `convertcase.c` program for converting a lowercase character found after a period in a file into uppercase is as follows:

```cpp
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFFSIZE 255

void main (int argc, char* argv[])
{
    FILE *fp;
    char buffer[BUFFSIZE];
    int i,n;

    fp = fopen (argv [1],"r");
    if (fp == NULL) {
        printf("%s file does not exist\n", argv[1]);
        exit(1);
    }
    while (!feof(fp))
    {
        fgets(buffer, BUFFSIZE, fp);
        n=strlen(buffer);
        for(i=0;i<n;i++)
        {
            if(buffer[i]=='.')
            {
                i++;
                while(buffer[i]==' ')
                {
                    i++;
                }
                if(buffer[i] >=97 && buffer[i] <=122)
                {
                    buffer[i]=buffer[i]-32;
                }
            }
        }
        puts(buffer);
    }
    fclose(fp);
}
```

Now, let's go behind the scenes.

# How it works...

The file whose name is supplied as a command-line argument is opened in read-only mode and is pointed to by the file pointer, `fp`. This recipe is focused on reading a file and changing its case, so if the file does not exist or does not have read permission, an error will be displayed and the program will terminate.

A `while` loop will be set to execute until `feof` (the end of file) is reached. Within the `while` loop, each line of the file will be read one by one and assigned to the string named `buffer`. The `fgets()` function will be used to read one line at a time from the file. A number of characters will be read from the file until the newline character, `\n`, is reached, to a maximum of 254.

The following steps will be performed on each of the lines assigned to the string buffer:

1.  The length of the buffer string will be computed and a `for` loop will be executed to access each of the characters in the string buffer.
2.  The string buffer will be checked to see whether there are any periods in it.
3.  If one is found, the character following it will be checked to see whether it is into lowercase. ASCII values will be used to then convert the lowercase characters into uppercase (refer to [Chapter 2](d3f17a83-f91b-4831-81b4-107b3eb19092.xhtml), *Managing Strings*, for more information on the ASCII values that correspond to the letters of the alphabet). If the character following the period is in lowercase, a value of `32` will be deducted from the ASCII value of the lowercase character to convert it into uppercase. Remember, the ASCII value of uppercase characters is lower by a value of `32` than their corresponding lowercase characters.
4.  The updated string `buffer` with the character following the period converted into uppercase will be displayed on the screen.

When all the lines of the file are read and displayed, the file pointed to by the `fp` pointer will be closed.

Let's use GCC to compile the `convertcase.c` program as follows:

```cpp
D:\CBook>gcc convertcase.c -o convertcase
```

If you get no errors or warnings, this means that the `convertcase.c` program has been compiled into an executable file, `convertcase.exe`.

Let's say that I have created a file called `textfile.txt` with the following content:

```cpp
D:\CBook>type textfile.txt
I am trying to create a sequential file. it is through C programming. It is very hot today. I have a cat. do you like animals? It might rain. Thank you. Bye
```

The preceding command is executed in Windows' Command Prompt.

Let's run the executable file, `convertcase.exe`, and then pass the `textfile.txt` file to it, as shown in the following code:

```cpp
D:\CBook>./convertcase textfile.txt
I am trying to create a sequential file. It is through C programming. It is very hot today. I have a cat. Do you like animals? It might rain. Thank you. Bye
```

You can see in the preceding output that the characters that were in lowercase after the period are now converted into uppercase.

Let's move on to the next recipe!

# Displaying the contents of a random file in reverse order

Let's say that we have a random file that contains some lines of text. Let's find out how to reverse the contents of this file.

This program will not give the correct output if a random file does not exist. Please read *Appendix A* to learn how to create a random file.

# How to do it…

1.  Open the random file in read-only mode using the following code:

```cpp
fp = fopen (argv[1], "rb");
```

2.  If the file does not exist or does not have enough permissions, an error message will be displayed and the program will terminate, as shown in the following code:

```cpp
if (fp == NULL) {
     perror ("An error occurred in opening the file\n");
     exit(1);
 }
```

3.  To read the random file in reverse order, execute a loop equal to the number of lines in the file. Every iteration of the loop will read one line beginning from the bottom of the file. The following formula will be used to find out the number of lines in the file:

*total number of bytes used in the file/size of one line in bytes*

The code for doing this is as follows:

```cpp
fseek(fp, 0L, SEEK_END);
n = ftell(fp);
nol=n/sizeof(struct data);
```

4.  Because the file has to be read in reverse order, the file pointer will be positioned at the bottom of the file, as shown in the following code:

```cpp
fseek(fp, -sizeof(struct data)*i, SEEK_END); 
```

5.  Set a loop to execute that equals the number of lines in the file computed in step 3, as shown in the following code:

```cpp
for (i=1;i<=nol;i++)
```

6.  Within the loop, the file pointer will be positioned as follows:

![](img/99311de4-d388-4a5f-b870-cab7ceffb01c.png)

Figure 5.2

7.  To read the last line, the file pointer will be positioned at the byte location where the last line begins, at the **-1 x sizeof(line)** byte location. The last line will be read and displayed on the screen, as shown in the following code:

```cpp
fread(&line,sizeof(struct data),1,fp);
puts(line.str);
```

8.  Next, the file pointer will be positioned at the byte location from where the second last line begins, at the **-2 x sizeof(line)** byte location. Again, the second last line will be read and displayed on the screen.
9.  The procedure will be repeated until all of the lines in the file have been read and displayed on the screen.

The `readrandominreverse.c` program for reading the random file in reverse order is as follows:

```cpp
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct data{  
    char str[ 255 ];  
};

void main (int argc, char* argv[])
{
    FILE *fp;
    struct data line;
    int n,nol,i;
    fp = fopen (argv[1], "rb");
    if (fp == NULL) {
        perror ("An error occurred in opening the file\n");
        exit(1);
    }
    fseek(fp, 0L, SEEK_END); 
    n = ftell(fp);
    nol=n/sizeof(struct data);
    printf("The content of random file in reverse order is :\n");
    for (i=1;i<=nol;i++)
    {
        fseek(fp, -sizeof(struct data)*i, SEEK_END); 
        fread(&line,sizeof(struct data),1,fp);
        puts(line.str);
    }
    fclose(fp);
}
```

Now, let's go behind the scenes.

# How it works...

We will open the chosen file in read-only mode. If the file opens successfully, it will be pointed at by the file pointer `fp`. Next, we will find out the total number of lines in the file using the following formula:

*total number of bytes used by the file/number of bytes used by one line*

To know the total number of bytes used by the file, the file pointer will be positioned at the bottom of the file and we will invoke the `ftell` function. The `ftell` function finds the current location of the file pointer. Because the file pointer is at the end of the file, using this function will tell us the total number of bytes used by the file. To find the number of bytes used by one line, we will use the `sizeof` function. We will apply the preceding formula to compute the total number of lines in the file; this will be assigned to the variable, `nol`.

We will set a `for` loop to execute for `nol` number of times. Within the `for` loop, the file pointer will be positioned at the end of the last line so that all of the lines from the file can be accessed in reverse order. So, the file pointer is first set at the `(-1 * size of one line)` location at the bottom of the file. Once the file pointer is positioned at this location, we will use the `fread` function to read the last line of the file and assign it to the structure variable line. The string in line will then be displayed on the screen.

After displaying the last line on the screen, the file pointer will be set at the byte position of the second last line at `(-2 * size of one line)`. We will again use the `fread` function to read the second last line and display it on the screen.

This procedure will be executed for the number of times that the `for` loop executes, and the `for` loop will execute the same number of times as there are lines in the file. Then the file will be closed.

Let's use GCC to compile the `readrandominreverse.c` program, as follows:

```cpp
D:\CBook>gcc readrandominreverse.c -o readrandominreverse
```

If you get no errors or warnings, this means that the `readrandominreverse.c` program has been compiled into an executable file, `readrandominreverse.exe`.

Let's assume that we have a random file, `random.data`, with the following text:

```cpp
This is a random file. I am checking if the code is working
perfectly well. Random file helps in fast accessing of
desired data. Also you can access any content in any order.
```

Let's run the executable file, `readrandominreverse.exe`, to display the random file, `random.data`, in reverse order using the following code:

```cpp
D:\CBook>./readrandominreverse random.data
The content of random file in reverse order is :
desired data. Also you can access any content in any order.
perfectly well. Random file helps in fast accessing of
This is a random file. I am checking if the code is working
```

By comparing the original file with the preceding output, you can see that the file content is displayed in reverse order.

Now, let's move on to the next recipe!

# Counting the number of vowels in a file

In this recipe, we will open a sequential text file and count the number of vowels (both uppercase and lowercase) that it contains.

In this recipe, I will assume that a sequential file already exists. Please read *Appendix A* to learn how to create a sequential file. 

# How to do it…

1.  Open the sequential file in read-only mode using the following code:

```cpp
fp = fopen (argv [1],"r");
```

2.  If the file does not exist or does not have enough permissions, an error message will be displayed and the program will terminate, as shown in the following code:

```cpp
if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
 }
```

3.  Initialize the counter that will count the number of vowels in the file to `0`, as shown in the following code:

```cpp
count=0;
```

4.  One line is read from the file, as shown in the following code:

```cpp
fgets(buffer, BUFFSIZE, fp);
```

5.  Each character of the line is accessed and checked for any lowercase or uppercase vowels, as shown in the following code:

```cpp
if(buffer[i]=='a' || buffer[i]=='e' || buffer[i]=='i' || buffer[i]=='o' || buffer[i]=='u' || buffer[i]=='A' || buffer[i]=='E' || buffer[i]=='I' || buffer[i]=='O' || buffer[i]=='U')
```

6.  If any vowel is found, the value of the counter is incremented by `1`, as shown in the following code:

```cpp
count++;
```

7.  Step 5 will be repeated until the end of the line has been reached. Check whether the end of the file has been reached. Repeat from step 4 until the end of the file, as shown in the following code:

```cpp
while (!feof(fp))
```

8.  Display the count of the number of vowels in the file by printing the value in the counter variable on the screen, as shown in the following code:

```cpp
printf("The number of vowels are %d\n",count);
```

The preceding steps are shown in the following diagram:

![](img/80e5fba9-c43e-4dc4-b5b8-914ea63abac2.png)

Figure 5.3

The `countvowels.c` program to count the number of vowels in a sequential text file is as follows:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFSIZE 255

void main (int argc, char* argv[])
{
    FILE *fp;
    char buffer[BUFFSIZE];
    int n, i, count=0;
    fp = fopen (argv [1],"r");
    if (fp == NULL) {
        printf("%s file does not exist\n", argv[1]);
        exit(1);
    }
    printf("The file content is :\n");
    while (!feof(fp))
    {
        fgets(buffer, BUFFSIZE, fp);
        puts(buffer);
        n=strlen(buffer);
        for(i=0;i<n;i++)
        {
            if(buffer[i]=='a' || buffer[i]=='e' || buffer[i]=='i' || 
            buffer[i]=='o' || buffer[i]=='u' || buffer[i]=='A' || 
            buffer[i]=='E' || buffer[i]=='I' || buffer[i]=='O' || 
            buffer[i]=='U') count++;
        }         
    }
    printf("The number of vowels are %d\n",count);
    fclose(fp);
}
```

Now, let's go behind the scenes.

# How it works...

We will open the chosen sequential file in read-only mode. If the file opens successfully, it will be pointed at by the file pointer, `fp`. To count the number of vowels in the file, we will initialize a counter from `0`.

We will set a `while` loop to execute until the file pointer, `fp`, reaches the end of the file. Within the `while` loop, each line in the file will be read using the `fgets` function. The `fgets` function will read the `BUFFSIZE` number of characters from the file. The value of the `BUFFSIZE` variable is `255`, so `fgets` will read either `254` characters from the file or will read characters until the newline character, `\n`, is reached, whichever comes first.

The line read from the file is assigned to the `buffer `string. To display the file contents along with the count of the vowels, the content in the `buffer` string is displayed on the screen. The length of the `buffer` string will be computed and a `for` loop will be set to execute equaling the length of the string.

Each of the characters in the buffer string will be checked in the `for` loop. If any lowercase or uppercase vowels appear in the line, the value of the counter variable will be incremented by `1`. When the `while` loop ends, the counter variable will have the total count of the vowels present in the file. Finally, the value in the counter variable will be displayed on the screen.

Let's use GCC to compile the `countvowels.c` program as follows:

```cpp
D:\CBook>gcc countvowels.c -o countvowels
```

If you get no errors or warnings, then this means that the `countvowels.c` program has been compiled into an executable file called `countvowels.exe`.

Let's assume that we have a text file called `textfile.txt` with some content. We will run the executable file, `countvowels.exe`, and supply the `textfile.txt` file to it to count the number of vowels in it, as shown in the following code:

```cpp
D:\CBook>./countvowels textfile.txt
The file content is :
I am trying to create a sequential file. it is through C programming. It is very hot today. I have a cat. do you like animals? It might rain. Thank you. bye
The number of vowels are 49
```

You can see from the output of the program that the program not only displays the count of the vowels, but also the complete content of the file.

Now, let's move on to the next recipe!

# Replacing a word in a file with another word

Let's say that you want to replace all occurrences of the word `is` with the word `was` in one of your files. Let's find out how to do this.

In this recipe, I will assume that a sequential file already exists. Please read *Appendix A* to learn how to create a sequential file. 

# How to do it…

1.  Open the file in read-only mode using the following code:

```cpp
    fp = fopen (argv [1],"r");
```

2.  If the file does not exist or does not have enough permissions, an error message will be displayed and the program will terminate, as shown in the following code:

```cpp
if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
 }
```

3.  Enter the word to be replaced using the following code:

```cpp
printf("Enter a string to be replaced: ");
scanf("%s", str1);
```

4.  Enter the new word that will replace the old word using the following code:

```cpp
printf("Enter the new string ");
scanf("%s", str2);
```

5.  Read a line from the file using the following code:

```cpp
fgets(line, 255, fp);
```

6.  Check whether the word to be replaced appears anywhere in the line using the following code:

```cpp
if(line[i]==str1[w])
{
     oldi=i;
     while(w<ls1)
     {
         if(line[i] != str1[w])
             break;
         else
         {
             i++;
             w++;
         }
     }
}
```

7.  If the word appears in the line, then simply replace it with the new word using the following code:

```cpp
if(w==ls1)
{
     i=oldi;
     for (k=0;k<ls2;k++)
     {
         nline[x]=str2[k];
         x++;
     }
     i=i+ls1-1;
 }
```

8.  If the word does not appear anywhere in the line, then move on to the next step. Print the line with the replaced word using the following code:

```cpp
puts(nline);
```

9.  Check whether the end of the file has been reached using the following code:

```cpp
while (!feof(fp))
```

10.  If the end of the file has not yet been reached, go to step 4. Close the file using the following code:

```cpp
fclose(fp);
```

The `replaceword.c` program replaces the specified word in a file with another word and displays the modified content on the screen:

```cpp
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void main (int argc, char* argv[])
{
    FILE *fp;
    char line[255], nline[300], str1[80], str2[80];
    int i,ll, ls1,ls2, x,k, w, oldi;

    fp = fopen (argv [1],"r");
    if (fp == NULL) {
        printf("%s file does not exist\n", argv[1]);
        exit(1);
    }
    printf("Enter a string to be replaced: ");
    scanf("%s", str1);
    printf("Enter the new string ");
    scanf("%s", str2);
    ls1=strlen(str1);
    ls2=strlen(str2);
    x=0;
    while (!feof(fp))
    {
        fgets(line, 255, fp);
        ll=strlen(line);
        for(i=0;i<ll;i++)
        {
            w=0;
            if(line[i]==str1[w])
            {
                oldi=i;    
                while(w<ls1)
                {
                    if(line[i] != str1[w])
                        break;
                    else
                    {
                        i++;
                        w++;
                    }
                }
                if(w==ls1)
                {
                    i=oldi;
                    for (k=0;k<ls2;k++)
                    {
                        nline[x]=str2[k];
                        x++;
                    }
                    i=i+ls1-1;
                }
                else
                {
                    i=oldi;
                    nline[x]=line[i];
                    x++;       
                }         
            }
            else
            {
                nline[x]=line[i];
                x++;
            }
        }
        nline[x]='\0';
        puts(nline);
     }
     fclose(fp);
}
```

Now, let's go behind the scenes.

# How it works...

Open the chosen file in read-only mode. If the file opens successfully, then the file pointer, `fp`, will be set to point at it. Enter the word to be replaced and assign it to the string variable, `str1`. Similarly, enter the new string that will be assigned to another string variable, `str2`. The length of the two strings, `str1` and `str2`, will be computed and assigned to the variables, `ls1` and `ls2`, respectively.

Set a `while` loop to execute until the file pointed at by `fp` pointer gets over. Within the `while` loop, one line from the file will be read using the `fgets` function. The `fgets` function reads the file until the maximum length that is specified or the new line character, `\n`, is reached, whichever comes first. Because strings are terminated with a mandatory null character, `\0`, a maximum of `254` characters will be read from the file.

The string that is read from the file will be assigned to the `line` variable. The length of the `line` string will be computed and assigned to the `ll` variable. Using a `for` loop, each of the characters in the line variable will be accessed to check whether they match with `str1[0]`—that is, with the first character of the string to be replaced. The characters in the `line` variable that don't match with the string to be replaced will be assigned to another string, called `nline`. The `nline` string will contain the desired content—that is, all of the characters of the `line` variable and the new string. If it exists in `line`, then the string will be replaced with the new string and the entire modified content will be assigned to the new string, `nline`.

If the first character of the string to be replaced matches with any of the characters in `line`, then the `while` loop will be used to match all of the successive characters of the string that is to be replaced with the successive characters in `line`. If all of the characters of the string that is to be replaced match with successive characters in `line`, then all of the characters of the string to be replaced are replaced with the new string and assigned to the new string, `nline`. That way, the `while` loop will read one line of text at a time from the file, searching for occurrences of the string to be replaced. If it is found, it replaces it with the new string and assigns the modified line of text to another string, `nline`. The null character, `\0`, is added to the modified string, `nline`, and is displayed on the screen. Finally, the file pointed to by the file pointer, `fp`, is closed.

In this recipe, I am replacing the desired word and another string and displaying the updated content on the screen. If you want the updated content to be written into another file, then you can always open another file in write mode and execute the `fputs` function to write the updated content in it.

Let's use GCC to compile the `replaceword.c` program, as follows:

```cpp
D:\CBook>gcc replaceword.c -o replaceword
```

If you get no errors or warnings, then this means that the `replaceword.c` program  has been compiled into an executable file, `replaceword.exe`. Let's run the executable file, `replaceword.exe`, and supply a text file to it. We will assume that a text file called `textfile.txt` exists and has the following content:

```cpp
I am trying to create a sequential file. it is through C programming. It is very hot today. I have a cat. do you like animals? It might rain. Thank you. bye
```

Now, let's use this file to replace one of its words with another word using the following code:

```cpp
D:\CBook>./replaceword textfile.txt
Enter a string to be replaced: is
Enter the new string was
I am trying to create a sequential file. it was through C programming. It was very hot today. I have a cat. do you like animals? It might rain. Thank you. Bye
```

You can see that all occurrences of the word `is` are replaced by `was` in `textfile.txt`, and the modified content is displayed on the screen. We've successfully replaced the words of our choice.

Now, let's move on to the next recipe!

# Encrypting a file

Encryption means converting content into a coded format so that unauthorized persons will be unable to see or access the original content of the file. A text file can be encrypted by applying a formula to the ASCII value of the content.

The formula or code can be of your choosing, and it can be as simple or complex as you want. For example, let's say that you have chosen to replace the current ASCII value of all letters by moving them forward 15 values. In this case, if the letter is a lowercase *a* that has the ASCII value of 97, then the forward shift of the ASCII values by 15 will make the *encrypted* letter a lowercase *p*, which has the ASCII value of 112 (97 + 15 = 112).

In this recipe, I assume that a sequential file that you want to encrypt already exists. Please read *Appendix A* to learn how to create a sequential file. You can also refer to *Appendix A* if you want to know how an encrypted file is decrypted.

# How to do it…

1.  Open the source file in read-only mode using the following code:

```cpp
fp = fopen (argv [1],"r");
```

2.  If the file does not exist or does not have enough permissions, an error message will be displayed and the program will terminate, as shown in the following code:

```cpp
if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
 }
```

3.  Open the destination file, the file where the encrypted text will be written, in write-only mode using the following code:

```cpp
fq = fopen (argv[2], "w");
```

4.  Read a line from the file and access each of its characters using the following code:

```cpp
fgets(buffer, BUFFSIZE, fp);
```

5.  Using the following code, subtract a value of `45` from the ASCII value of each of the characters in the line to encrypt that character:

```cpp
for(i=0;i<n;i++)
    buffer[i]=buffer[i]-45;
```

6.  Repeat step 5 until the line is over. Once all of the characters in the line are encrypted, write the encrypted line into the destination file using the following code:

```cpp
fputs(buffer,fq);
```

7.  Check whether the end of the file has been reached using the following code:

```cpp
while (!feof(fp))
```

8.  Close the two files using the following code:

```cpp
fclose (fp);
fclose (fq);
```

The preceding steps are shown in the following diagram:

![](img/80c9c7ef-b0c4-4dff-9d33-ef97dd70f8ec.png)

Figure 5.4

The `encryptfile.c` program to encrypt a file is as follows:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#define BUFFSIZE 255
void main (int argc, char* argv[])
{
    FILE *fp,*fq;
    int  i,n;
    char buffer[BUFFSIZE];

    /* Open the source file in read mode */
    fp = fopen (argv [1],"r");
    if (fp == NULL) {
        printf("%s file does not exist\n", argv[1]);
        exit(1);
    }
    /* Create the destination file.  */
    fq = fopen (argv[2], "w");
    if (fq == NULL) {
        perror ("An error occurred in creating the file\n");
        exit(1);
    }
    while (!feof(fp))
    {
        fgets(buffer, BUFFSIZE, fp);
        n=strlen(buffer);
        for(i=0;i<n;i++)
            buffer[i]=buffer[i]-45;
        fputs(buffer,fq);
    }
    fclose (fp);
    fclose (fq); 
}
```

Now, let's go behind the scenes.

# How it works...

The first file name that is passed through the command-line arguments is opened in read-only mode. The second file name that is passed through the command-line arguments is opened in write-only mode. If both files are opened correctly, then the `fp` and `fq` pointers , respectively, will point at the read-only and write-only files.

We will set a `while` loop to execute until it reaches the end of the source file. Within the loop, one line from the source file will be read using the `fgets` function. The `fgets` function reads the specified number of bytes from the file or until the new line character, `\n`, is reached. If the new line character does not appear in the file, then the `BUFFSIZE` constant limits the bytes to be read from the file to `254`.

The line read from the file is assigned to the `buffer` string . The length of the string `buffer` is computed and assigned to the variable, `n`. We will then set a `for` loop to execute until it reaches the end of the length of the `buffer` string, and within the loop, the ASCII value of each character will be changed.

To encrypt the file, we will subtract a value of `45 `from the ASCII value of each of the characters, although we can apply any formula we like. Just ensure that you remember the formula, as we will need to reverse it in order to decrypt the file.

After applying the formula to all of the characters, the encrypted line will be written into the target file. In addition, to display the encrypted version on the screen, the encrypted line will be displayed on the screen.

When the `while` loop is finished, all of the lines from the source file will be written into the target file after they are encrypted. Finally, the two files will be closed.

Let's use GCC to compile the `encryptfile.c` program, as follows:

```cpp
D:\CBook>gcc encryptfile.c -o encryptfile
```

If you get no errors or warnings, this means that the `encryptfile.c` program  has been compiled into an executable file, `encryptfile.exe`. Let's run this executable file.

Before running the executable file, though, let's take a look at the text file, `textfile.txt`, which will be encrypted using this program. The contents of this text file are as follows:

```cpp
I am trying to create a sequential file. it is through C programming. It is very hot today. I have a cat. do you like animals? It might rain. Thank you. bye
```

Let's run the executable file, `encryptfile.exe`, on `textfile.txt` and put the encrypted content into another file named `encrypted.txt` using the following code:

```cpp
D:\CBook>./encryptfile textfile.txt encrypted.txt
```

The normal content in `textfile.txt` is encrypted and the encrypted content is written into another file named `encrypted.txt`. The encrypted content will appear as follows:

```cpp
D:\CBook>type encrypted.txt
≤4@≤GEL<A:≤GB≤6E84G8≤4≤F8DH8AG<4?≤9<?8≤<G≤<F≤G;EBH:;≤≤CEB:E4@@<A:≤≤≤G≤<F≤I8EL≤;BG≤GB74L≤;4I8≤4≤64G≤≤7B≤LBH≤?<>8≤4A<@4?F≤≤≤≤G≤@<:;G≤E4<A';4A>≤LBH≤5L8
```

The preceding command is executed in Windows' Command Prompt.

Voila! We've successfully encrypted the file!