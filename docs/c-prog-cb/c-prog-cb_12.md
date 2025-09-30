# 附录 A

在本书的这一部分，我们将探讨一些超出第三章“探索函数”范围的其他菜谱：

+   创建一个顺序文件并将一些文本输入到它里面

+   从顺序文件读取内容并在屏幕上显示

+   创建一个随机文件并将一些数据输入到它里面

+   从随机文件读取内容并在屏幕上显示

+   解密加密文件的内容

# 创建一个顺序文件并将一些数据输入到它里面

在这个菜谱中，我们将创建一个顺序文件，用户可以输入任意数量的行到它里面。要创建的文件名将通过命令行参数传递。你可以输入任意多的行到文件中，完成时，你必须输入 `stop`，然后按 *Enter* 键。

# 如何操作…

1.  以只写模式打开一个顺序文件，并用文件指针指向它：

```cpp
fp = fopen (argv[1], "w");
```

1.  当提示时输入文件内容：

```cpp
printf("Enter content for the file\n");
gets(str);
```

1.  当你完成输入文件内容时，请输入 `stop`：

```cpp
while(strcmp(str, "stop") !=0)
```

1.  如果你输入的字符串不是 `stop`，则该字符串将被写入文件：

```cpp
fputs(str,fp);
```

1.  关闭文件指针以释放分配给文件的所有资源：

```cpp
fclose(fp);
```

创建顺序文件的 `createtextfile.c` 程序如下：

```cpp
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void main (int argc, char* argv[])
{
   char str[255];
   FILE *fp;

  fp = fopen (argv[1], "w");
  if (fp == NULL) {
     perror ("An error occurred in creating the file\n");
    exit(1);
  }
  printf("Enter content for the file\n");
  gets(str);
  while(strcmp(str, "stop") !=0){
      fputs(str,fp);
      gets(str);
  }
  fclose(fp);
}
```

现在，让我们幕后了解代码，以便更好地理解它。

# 它是如何工作的...

我们通过名称 `fp` 定义一个文件指针。我们将以只写模式打开通过命令行参数提供的顺序文件，并将 `fp` 文件指针设置为指向它。如果文件无法以只写模式打开，可能是因为权限不足或磁盘空间限制。将显示错误消息，程序将终止。

如果文件以只写模式成功打开，你将被提示输入文件内容。你输入的所有文本都将分配给 `str` 字符串变量，然后写入文件。完成输入文件内容后，你应该输入 `stop`。最后，我们将关闭文件指针。

让我们使用 GCC 编译 `createtextfile.c` 程序，如下所示：

```cpp
D:\CBook>gcc createtextfile.c -o createtextfile
```

如果你没有收到任何错误或警告，这意味着 `createtextfile.c` 程序已经被编译成了一个可执行文件，名为 `createtextfile.exe`。让我们运行这个可执行文件：

```cpp
D:\CBook>createtextfile textfile.txt
Enter content for the file
I am trying to create a sequential file. it is through C programming.   It is very hot today
I have a cat.  do you like animals?    It might rain
Thank you. bye
stop
```

哇！我们已经成功创建了一个顺序文件并在其中输入了数据。

现在让我们继续下一个菜谱！

# 从顺序文件读取内容并在屏幕上显示

在这个菜谱中，我们假设一个顺序文件已经存在，因此我们将读取该文件的内容并在屏幕上显示。我们想要读取内容的文件名将通过命令行参数提供。

# 如何操作...

1.  以只读模式打开顺序文件并将 `fp` 文件指针设置为指向它：

```cpp
fp = fopen (argv [1],"r");
```

1.  如果文件无法以只读模式打开，程序将终止：

```cpp
if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
}
```

1.  设置一个`while`循环，直到文件末尾：

```cpp
while (!feof(fp))
```

1.  在`while`循环中，一次读取一行文件并显示在屏幕上：

```cpp
fgets(buffer, BUFFSIZE, fp);
puts(buffer);
```

1.  关闭文件指针以释放分配给文件的所有资源：

```cpp
fclose(fp);
```

读取顺序文件的`readtextfile.c`程序如下：

```cpp
#include <stdio.h>
#include <stdlib.h>

#define BUFFSIZE 255

void main (int argc, char* argv[])
{
   FILE *fp;
   char buffer[BUFFSIZE];

  fp = fopen (argv [1],"r");
  if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
  }
  while (!feof(fp))
  {
    fgets(buffer, BUFFSIZE, fp);
    puts(buffer);
  }
  fclose(fp);
}
```

现在，让我们深入了解代码背后的情况。

# 它是如何工作的…

我们将定义一个名为`fp`的文件指针和一个名为`buffer`的字符串，大小为 255。我们将以只读模式打开通过命令行参数提供的顺序文件，并将`fp`文件指针设置为指向它。如果因为文件不存在或权限不足而无法以只读模式打开文件，将显示错误消息，程序将终止。

如果文件以只读模式成功打开，设置一个`while`循环，直到文件末尾。使用`fgets`函数逐行从文件中读取；从文件中读取的行被分配给`buffer`字符串。然后，在屏幕上显示`buffer`字符串中的内容。当到达文件末尾时，`while`循环将终止，文件指针被关闭以释放分配给文件的所有资源。

让我们使用 GCC 编译`readtextfile.c`程序，如下所示：

```cpp
D:\CBook>gcc readtextfile.c -o readtextfile
```

如果你没有错误或警告，这意味着`readtextfile.c`程序已经被编译成一个可执行文件，`readtextfile.exe`。假设我们想要读取内容的文件是 textfile.txt，让我们运行可执行文件`readtextfile.exe`：

```cpp
D:\CBook>readtextfile textfile.txt
I am trying to create a sequential file. it is through C programming.   It is very hot today. I have a cat.  do you like animals?    It might rain. Thank you. bye
```

哇！我们已经成功从我们的顺序文件中读取内容并在屏幕上显示。

现在，让我们继续到下一个菜谱！

# 创建一个随机文件并将一些数据输入其中

在这个菜谱中，我们将创建一个随机文件，并将一些文本行输入其中。随机文件是有结构的，随机文件中的内容是通过结构写入的。使用结构创建文件的好处是我们可以直接计算任何结构的定位，并且可以随机访问文件中的任何内容。要创建的文件名通过命令行参数传递。

# 如何做到这一点…

创建随机文件并在其中输入几行文本的步骤如下。你可以输入任意数量的行；完成时，只需按`stop`，然后按`Enter`键：

1.  定义一个由字符串成员组成的结构：

```cpp
struct data{
    char str[ 255 ];
};
```

1.  以只写模式打开一个随机文件，并用文件指针指向它：

```cpp
fp = fopen (argv[1], "wb");
```

1.  如果文件无法以只写模式打开，程序将终止：

```cpp
if (fp == NULL) {
    perror ("An error occurred in creating the file\n");
    exit(1);
}
```

1.  在提示时输入文件内容并将其存储到结构成员中：

```cpp
printf("Enter file content:\n");
gets(line.str);
```

1.  如果输入的文本不是`stop`，则包含该文本的结构将被写入文件：

```cpp
while(strcmp(line.str, "stop") !=0){
    fwrite( &line, sizeof(struct data), 1, fp );
```

1.  重复步骤 4 和 5，直到你输入`stop`。

1.  当你输入`stop`时，指向文件的文件指针被关闭以释放分配给文件的资源：

```cpp
fclose(fp);
```

创建随机文件的`createrandomfile.c`程序如下：

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
    fp = fopen (argv[1], "wb");
    if (fp == NULL) {
        perror ("An error occurred in creating the file\n");            
        exit(1);
    }
    printf("Enter file content:\n");
    gets(line.str);
    while(strcmp(line.str, "stop") !=0){
        fwrite( &line, sizeof(struct data), 1, fp );
        gets(line.str);
    }
    fclose(fp);
}
```

现在，让我们幕后了解一下代码。

# 它是如何工作的…

让我们从定义一个名为`data`的结构体开始，它包含一个名为`str`的成员，这是一个大小为 255 的字符串变量。然后我们将定义一个名为`fp`的文件指针，接着是一个类型为`data`结构体的变量`line`，这样行就变成了一个包含名为`str`的成员的结构体。我们将以只写模式打开一个名为通过命令行参数提供的随机文件，并将`fp`文件指针设置为指向它。如果由于任何原因无法以只写模式打开文件，将显示错误消息，程序将终止。

你将被提示输入文件内容。你输入的文本将被分配给行结构体的`str`成员。因为你应该输入`stop`来表示你已完成文件中的数据输入，所以你输入的文本将与`stop`字符串进行比较。如果输入的文本不是`stop`，它将被写入由`fp`文件指针指向的文件。

因为它是随机文件，所以文本通过结构行写入文件。`fwrite`函数将等于结构行大小的字节数写入由`fp`指针指向的文件在其当前位置。行结构中的`str`成员中的文本被写入文件。当用户输入的文本是`stop`时，指向文件的文件指针`fp`被关闭。

让我们使用 GCC 编译`createrandomfile.c`程序，如下所示：

```cpp
D:\CBook>gcc createrandomfile.c -o createrandomfile
```

如果你没有错误或警告，这意味着`createrandomfile.c`程序已经被编译成一个可执行文件，`createrandomfile.exe`。假设我们想要创建一个名为`random.data`的随机文件，让我们运行可执行文件`createrandomfile.exe`：

```cpp
D:\CBook>createrandomfile random.data
Enter file content:
This is a random file. I am checking if the code is working
perfectly well. Random file helps in fast accessing of
desired data. Also you can access any content in any order.
stop
```

哇！我们已经成功创建了一个随机文件并在其中输入了一些数据。

现在让我们继续下一个菜谱！

# 从随机文件读取内容并在屏幕上显示

在这个菜谱中，我们将读取随机文件的内容，并将其显示在屏幕上。因为随机文件中的内容由记录组成，其中记录的大小已经知道，所以可以从随机文件中随机选择任何记录；因此，这种类型的文件被称为*随机文件*。要从随机文件中访问第*n*条记录，我们不需要先读取前*n*-1 条记录，就像在顺序文件中做的那样。我们可以计算该记录的位置，并直接访问它。要读取的文件名通过命令行参数传递。

# 如何做到这一点…

1.  定义一个包含字符串成员的结构体：

```cpp
struct data{
    char str[ 255 ];
};
```

1.  以只读模式打开一个随机文件，并用文件指针指向它：

```cpp
fp = fopen (argv[1], "rb");
```

1.  如果文件无法以只读模式打开，程序将终止：

```cpp
if (fp == NULL) {
    perror ("An error occurred in opening the file\n");
    exit(1);
}
```

1.  找到文件中的总字节数。将检索到的文件总字节数除以每条记录的大小，以获取文件中的总记录数：

```cpp
fseek(fp, 0L, SEEK_END); 
n = ftell(fp);
nol=n/sizeof(struct data);
```

1.  使用`for`循环一次读取文件中的一个记录：

```cpp
for (i=1;i<=nol;i++)
fread(&line,sizeof(struct data),1,fp);
```

1.  从随机文件读取的内容是通过步骤 1 中定义的结构体获取的。通过显示分配给结构体成员的文件内容来显示文件内容：

```cpp
puts(line.str);
```

1.  当`for`循环结束时，达到文件末尾。关闭文件指针以释放分配给文件的资源：

```cpp
fclose(fp);
```

读取随机文件内容的`readrandomfile.c`程序如下：

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
    rewind(fp);
    printf("The content in file is :\n");
    for (i=1;i<=nol;i++)
    {
        fread(&line,sizeof(struct data),1,fp);
        puts(line.str);
    }
    fclose(fp);
}
```

现在，让我们深入了解代码，以更好地理解它。

# 它是如何工作的...

我们将定义一个名为`data`的结构体，它包含一个名为`str`的成员，该成员是一个大小为 255 的字符串变量。然后，定义一个名为`fp`的文件指针和一个类型为数据结构`line`的变量，这样行就变成了一个具有名为`str`的成员的结构体。我们将以只读模式打开一个随机文件，其名称通过命令行参数提供，并将`fp`文件指针设置为指向它。如果由于任何原因无法以只读模式打开文件，将显示错误消息，并终止程序。如果引用了任何不存在的文件，或者文件没有足够的权限，则可能发生文件错误。如果文件成功以只读模式打开，下一步是找到文件中的记录总数。为此，应用以下公式：

*文件中的总字节数/每条记录的大小*

要找到文件中的总字节数，首先通过调用`fseek`函数将文件指针移动到文件末尾。然后，使用`ftell`函数检索文件消耗的总字节数。然后，我们将文件中的总字节数除以每条记录的大小，以确定文件中的总记录数。

现在，我们已经准备好一次读取文件中的一个记录，为此，我们将文件指针移动到文件的开头。我们将设置一个`for`循环，使其执行次数与文件中的记录数相同。在`for`循环内部，我们将调用`fread`函数一次读取文件中的一个记录。从文件中读取的文本被分配给行结构体的`str`成员。行结构体中`str`成员的内容将在屏幕上显示。当`for`循环结束时，由`fp`文件指针指向的文件将被关闭，以释放分配给文件的资源。

让我们使用 GCC 编译`readrandomfile.c`程序，如下所示：

```cpp
D:\CBook>gcc readrandomfile.c -o readrandomfile
```

如果没有错误或警告，这意味着`readrandomfile.c`程序已被编译成可执行文件，`readrandomfile.exe`。假设我们想要创建一个名为`random.data`的随机文件，让我们运行可执行文件，`readrandomfile.exe`：

```cpp
D:\CBook>readrandomfile random.data
The content in file is :
This is a random file. I am checking if the code is working
perfectly well. Random file helps in fast accessing of
desired data. Also you can access any content in any order.
```

哇！我们已经成功从随机文件中读取内容并在屏幕上显示它。

现在让我们继续下一个菜谱！

# 解密加密文件的内容

在这个菜谱中，我们将读取一个加密文件。我们将解密其内容，并将解密后的内容写入另一个顺序文件。两个文件名，加密文件和我们将保存解密版本的文件，都是通过命令行参数提供的。

# 如何操作...

在这个程序中使用了两个文件。一个是只读模式打开的，另一个是只写模式打开的。读取并解密一个文件的内容，并将解密内容存储在另一个文件中。以下是将现有加密文件解密并保存到另一个文件的步骤：

1.  以只读和只写模式打开两个文件：

```cpp
fp = fopen (argv [1],"r");
fq = fopen (argv[2], "w");
```

1.  如果任一文件无法以相应模式打开，程序将在显示错误消息后终止：

```cpp
if (fp == NULL) {
    printf("%s file does not exist\n", argv[1]);
    exit(1);
}
if (fq == NULL) {
    perror ("An error occurred in creating the file\n");
    exit(1);
}
```

1.  设置一个`while`循环来执行。它将逐行从要读取的文件中读取：

```cpp
while (!feof(fp))
fgets(buffer, BUFFSIZE, fp);
```

1.  从文件中读取的行的长度被计算：

```cpp
n=strlen(buffer);
```

1.  设置一个`for`循环来执行。这将逐个访问行的所有字符：

```cpp
for(i=0;i<n;i++)
```

1.  将`45`的值加到每个字符的 ASCII 值上以解密它。我假设每个字符的 ASCII 值减去`45`以加密文件：

```cpp
buffer[i]=buffer[i]+45;
```

1.  解密后的行将被写入第二个文件：

```cpp
fputs(buffer,fq);
```

1.  当`while`循环完成后（读取，即正在解密的文件），两个文件指针都将被关闭以释放分配给它们的资源：

```cpp
fclose (fp);
fclose (fq);
```

用于解密加密文件的`decryptfile.c`程序如下：

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
    fp = fopen (argv [1],"r");
    if (fp == NULL) {
        printf("%s file does not exist\n", argv[1]);
        exit(1);
    }
    fq = fopen (argv[2], "w");                    if (fq == NULL) {
        perror ("An error occurred in creating the file\n");
        exit(1);
    }
    while (!feof(fp))
    {
        fgets(buffer, BUFFSIZE, fp);
        n=strlen(buffer);
        for(i=0;i<n;i++)
            buffer[i]=buffer[i]+45;   
        fputs(buffer,fq);
    }
    fclose (fp);
    fclose (fq); 
}
```

现在，让我们深入了解代码，以更好地理解它。

# 它是如何工作的...

我们将定义两个文件指针，`fp`和`fq`。我们将以只读模式打开通过命令行参数提供的第一个文件，并将第二个文件以只写模式打开。如果无法以只读模式和只写模式分别打开文件，将显示错误消息，程序将终止。以只读模式打开的文件由`fp`文件指针指向，以只写模式打开的文件由`fq`文件指针指向。

我们将设置一个`while`循环来执行，该循环将逐行读取由`fp`指针指向的文件的每一行。`while`循环将继续执行，直到到达由`fp`指向的文件的末尾。

在 `while` 循环中，读取一行并将其分配给 `buffer` 字符串变量。计算行的长度。然后设置一个 `for` 循环，执行到行的末尾；也就是说，访问行的每个字符。我们将把值 `45` 添加到每个字符的 ASCII 值以加密它。之后，我们将解密后的行写入第二个文件。该文件由 `fq` 文件指针指向。当读取内容的文件完成时，关闭两个文件指针以释放分配给两个文件的资源。

让我们使用 GCC 编译 `decryptfile.c` 程序，如下所示：

```cpp
D:\CBook>gcc decryptfile.c -o decryptfile
```

假设加密文件命名为 `encrypted.txt`。让我们看看这个文件中的加密文本：

```cpp
D:\CBook>type encrypted.txt
≤4@≤GEL<A:≤GB≤6E84G8≤4≤F8DH8AG<4?≤9<?8≤<G≤<F≤G;EBH:;≤≤CEB:E4@@<A:≤≤≤G≤<F≤I8EL≤;BG≤GB74L≤;4I8≤4≤64G≤≤7B≤LBH≤?<>8≤4A<@4?F≤≤≤≤G≤@<:;G≤E4<A';4A>≤LBH≤5L8
```

上述命令是在 Windows 的命令提示符中执行的。

如果在编译文件时没有错误或警告，这意味着 `decryptfile.c` 程序已经被编译成一个可执行文件，名为 `decryptfile.exe`。假设已经存在一个名为 `encrypted.txt` 的加密文件，并且你想将其解密到另一个文件中，名为 `originalfile.txt`。因此，让我们运行可执行文件 `decryptfile.exe` 来解密 `encrypted.txt` 文件：

```cpp
D:\CBook>decryptfile encrypted.txt originalfile.txt
```

让我们查看 `orignalfile.txt` 的内容，看看它是否包含文件的解密版本：

```cpp
D:\CBook>type originalfile.txt
I am trying to create a sequential file. it is through C programming.   It is very hot today. I have a cat.  do you like animals?    It might rain. Thank you. bye
```

哇！你可以看到 `originalfile.txt` 包含了解密后的文件。
