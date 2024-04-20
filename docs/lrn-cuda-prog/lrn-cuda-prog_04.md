# 核心执行模型和优化策略

CUDA 编程有一个主机操作的过程。例如，我们需要分配全局内存，将数据传输到 GPU，执行核心函数，将数据传输回主机，清理全局内存。这是因为 GPU 是系统中的一个额外处理单元，所以我们需要关心它的执行和数据传输。这是与 CPU 编程相比另一个不同的 GPU 编程方面。

在本章中，我们将涵盖 CUDA 核心执行模型和 CUDA 流，它们控制 CUDA 操作。然后，我们将讨论系统级别的优化策略。接下来，我们将涵盖 CUDA 事件来测量 GPU 事件时间，以及如何使用 CUDA 事件来测量核心执行时间。之后，我们将涵盖各种 CUDA 核心执行模型，并讨论这些特性对 GPU 操作的影响。

本章将涵盖以下主题：

+   使用 CUDA 流的核心执行

+   流水线化 GPU 执行

+   CUDA 回调函数

+   具有优先级的 CUDA 流

+   使用 CUDA 事件估计核心执行时间

+   CUDA 动态并行性

+   网格级协作组

+   使用 OpenMP 的 CUDA 核心调用

+   多进程服务

+   核心执行开销比较

# 技术要求

本章要求我们使用的 CUDA 版本应该晚于 9.x，并且 GPU 架构应该是 Volta 或 Turing。如果你使用的是 Pascal 架构的 GPU，那么跳过*Grid-level cooperative groups*部分，因为这个特性是为 Volta 架构引入的。

# 使用 CUDA 流的核心执行

在 CUDA 编程中，流是与 GPU 相关的一系列命令。换句话说，所有的核心调用和数据传输都由 CUDA 流处理。默认情况下，CUDA 提供了一个默认流，所有的命令都隐式地使用这个流。因此，我们不需要自己处理这个。

CUDA 支持显式创建额外的流。虽然流中的操作是顺序的，但 CUDA 可以通过使用多个流同时执行多个操作。让我们学习如何处理流，以及它们具有哪些特性。

# CUDA 流的使用

以下代码展示了如何创建、使用和终止 CUDA 流的示例：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
foo_kernel<<< grid_size, block_size, 0, stream >>>();
cudaStreamDestroy(stream);
```

正如你所看到的，我们可以使用`cudaStream_t`来处理 CUDA 流。而且，我们可以使用`cudaStreamCreate()`来创建它，使用`cudaStreamDestroy()`来终止它。注意我们应该提供一个指向`cudaStreamCreate()`的指针。创建的流会传递给核心函数的第四个参数。

然而，我们之前并没有提供这样的流。这是因为 CUDA 提供了一个默认流，以便所有的 CUDA 操作都可以进行。现在，让我们编写一个使用默认流和多个流的应用程序。然后，我们将看到我们的应用程序如何改变。

首先，让我们编写一个使用默认 CUDA 流的应用程序，如下所示：

```cpp
__global__ void foo_kernel(int step)
{
    printf("loop: %d\n", step);
}

int main()
{
    for (int i = 0; i < 5; i++)
 // CUDA kernel call with the default stream
 foo_kernel<<< 1, 1, 0, 0 >>>(i);
    cudaDeviceSynchronize();
    return 0;
}
```

正如你在代码中看到的，我们以流 ID 为`0`调用了核心函数，因为默认流的标识值为`0`。编译代码并查看执行输出：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_default_stream ./1_cuda_default_stream.cu
```

输出是什么？我们可以期待输出将是循环索引的顺序。以下时间轴视图显示了这段代码的操作：

![](img/870c3559-e2cf-4a27-a481-53697923141d.png)

可以预期，在同一个流中进行循环操作将显示核心执行的顺序。那么，如果我们使用多个 CUDA 流，并且每个循环步骤使用不同的流，会有什么改变？以下代码展示了使用不同流从 CUDA 核心函数打印循环索引的示例：

```cpp
__global__ void foo_kernel(int step)
{
    printf("loop: %d\n", step);
}

int main()
{
    int n_stream = 5;
    cudaStream_t *ls_stream;
    ls_stream = (cudaStream_t*) new cudaStream_t[n_stream];

    // create multiple streams
    for (int i = 0; i < n_stream; i++)
        cudaStreamCreate(&ls_stream[i]);

    // execute kernels with the CUDA stream each
    for (int i = 0; i < n_stream; i++)
        foo_kernel<<< 1, 1, 0, ls_stream[i] >>>(i);

    // synchronize the host and GPU
    cudaDeviceSynchronize();

    // terminates all the created CUDA streams
    for (int i = 0; i < n_stream; i++)
        cudaStreamDestroy(ls_stream[i]);
    delete [] ls_stream;

    return 0;
}
```

在这段代码中，我们有五个调用，与之前的代码相同，但这里我们将使用五个不同的流。为此，我们建立了一个`cudaStream_t`数组，并为每个流创建了流。你对这个改变有什么期待？打印输出将与之前的版本相同。运行以下命令来编译这段代码：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_mutli_stream ./2_cuda_multi_stream.cu
```

然而，这并不保证它们具有相同的操作。正如我们在开始时讨论的，这段代码展示了多个流的并发性，如下面的截图所示：

![](img/c34f9def-ab64-4d9e-a159-f9ecb23c807d.png)

正如你在截图底部所看到的，五个独立的流同时执行相同的内核函数，并且它们的操作相互重叠。由此，我们可以得出流的两个特点，如下所示：

1.  内核执行与主机是异步的。

1.  不同流中的 CUDA 操作是彼此独立的。

利用流的并发性，我们可以通过重叠独立操作来获得额外的优化机会。

# 流级别的同步

CUDA 流提供了流级别的同步，使用`cudaStreamSynchronize()`函数。使用这个函数会强制主机等待直到某个流的操作结束。这为我们迄今为止使用的`cudaDeviceSynchronize()`函数提供了重要的优化。

我们将在接下来的部分讨论如何利用这一特性，但让我们在这里讨论它的基本操作。前面的例子展示了在循环中没有同步的并发操作。然而，我们可以通过使用`cudaStreamSynchronize()`函数来阻止主机执行下一个内核执行。下面的代码展示了在内核执行结束时使用流同步的示例：

```cpp
// execute kernels with the CUDA stream each
for (int i = 0; i < n_stream; i++) {
   foo_kernel<<< 1, 1, 0, ls_stream[i] >>>(i);
   cudaStreamSynchronize(ls_stream[i]);
}
```

我们可以很容易地预测，由于同步，内核操作的并发性将消失。为了确认这一点，让我们对此进行分析，看看这对内核执行的影响：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_mutli_stream_with_sync ./3_cuda_multi_stream_with_sync.cu
```

下面的截图显示了结果：

![](img/f9a4b008-4225-4766-ac33-c26bec6a0fd6.png)

正如你所看到的，所有的内核执行没有重叠点，尽管它们是用不同的流执行的。利用这一特性，我们可以让主机等待特定流操作的开始和结果。

# 使用默认流

为了让多个流同时运行，我们应该使用我们显式创建的流，因为所有流操作都与默认流同步。下面的截图显示了默认流的同步操作效果：

![](img/e31e5516-2fb9-434c-a0c7-9d729fbdcde0.png)

我们可以通过修改我们的多流内核调用操作来实现这一点，就像这样：

```cpp
for (int i = 0; i < n_stream; i++)
    if (i == 3)
        foo_kernel<<< 1, 1, 0, 0 >>>(i);
    else
        foo_kernel<<< 1, 1, 0, ls_stream[i] >>>(i);
```

运行以下命令编译代码：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_multi_stream_with_default ./4_cuda_multi_stream_with_default.cu
```

因此，我们可以看到最后一个操作无法与前面的内核执行重叠，而是必须等到第四个内核执行完成后才能进行。

# GPU 执行的流水线

多个流的主要好处之一是将数据传输与内核执行重叠。通过重叠内核操作和数据传输，我们可以隐藏数据传输开销并提高整体性能。

# GPU 流水线的概念

当我们执行内核函数时，我们需要将数据从主机传输到 GPU，然后将结果从 GPU 传输回主机。下面的图表显示了在主机和内核执行之间传输数据的迭代操作的示例：

![](img/d8db7a9f-522f-4e8f-aa07-19efc6233fc8.png)

然而，内核执行基本上是异步的，主机和 GPU 可以同时运行。如果主机和 GPU 之间的数据传输具有相同的特性，我们就能够重叠它们的执行，就像我们在前面的部分中看到的那样。下面的图表显示了当数据传输可以像正常的内核操作一样执行，并与流一起处理时的操作：

![](img/3b531d26-ec42-4576-9b9c-511bd175b612.png)

在这个图表中，我们可以看到主机和设备之间的数据传输可以与内核执行重叠。然后，这种重叠操作的好处是减少应用程序的执行时间。通过比较两张图片的长度，您将能够确认哪个操作的吞吐量更高。

关于 CUDA 流，所有 CUDA 操作——数据传输和内核执行——在同一个流中是顺序的。然而，它们可以与不同的流同时操作。以下图表显示了多个流的重叠数据传输和内核操作：

![](img/c3141f46-c05d-4e22-85b4-bd4a2c193f9a.png)

为了实现这样的流水线操作，CUDA 有三个先决条件：

1.  主机内存应该分配为固定内存——CUDA 提供了`cudaMallocHost()`和`cudaFreeHost()`函数来实现这一目的。

1.  在主机和 GPU 之间传输数据而不阻塞主机——CUDA 提供了`cudaMemcpyAsync()`函数来实现这一目的。

1.  管理每个操作以及不同的 CUDA 流，以实现并发操作。

现在，让我们编写一个简单的应用程序来对工作负载进行流水线处理。

# 构建流水线执行

以下代码显示了异步数据传输的片段以及在执行结束时 CUDA 流的同步：

```cpp
cudaStream_t stream;
float *h_ptr, *d_ptr;    size_t byte_size = sizeof(float) * BUF_SIZE;

cudaStreamCreate(&stream);               // create CUDA stream
cudaMallocHost(h_ptr, byte_size);        // allocates pinned memory
cudaMalloc((void**)&d_ptr, byte_size);   // allocates a global memory

// transfer the data from host to the device asynchronously
cudaMemcpyAsync(d_ptr, h_ptr, byte_size, cudaMemcpyHostToDevice, stream);

... { kernel execution } ...

// transfer the data from the device to host asynchronously
cudaMemcpyAsync(h_ptr, d_ptr, byte_size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

// terminates allocated resources
cudaStreamDestroy(stream);
cudaFree(d_ptr);
cudaFreeHost(h_ptr);
```

这段代码展示了如何分配固定内存，并使用用户创建的流传输数据。通过合并这个例子和多个 CUDA 流操作，我们可以实现流水线 CUDA 操作。

现在，让我们构建一个应用程序，其中包含数据传输和内核执行的流水线操作。在这个应用程序中，我们将使用一个将两个向量相加的内核函数，通过切片流的数量，并输出其结果。然而，内核的实现在主机代码级别不需要任何更改。但是，我们将迭代加法操作 500 次以延长内核执行时间。因此，实现的内核代码如下：

```cpp
__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}
```

为了处理每个流的操作，我们将创建一个管理 CUDA 流和 CUDA 操作的类。这个类将允许我们管理 CUDA 流以及索引。以下代码显示了该类的基本架构：

```cpp
class Operator
{
private:
    int index;

public:
    Operator() {
        cudaStreamCreate(&stream);    // create a CUDA stream
    }

    ~Operator() {
        cudaStreamDestroy(stream);    // terminate the CUDA stream
    }

    cudaStream_t stream;
    void set_index(int idx) { index = idx; }
    void async_operation(float *h_c, const float *h_a, 
                         const float *h_b,
                         float *d_c, float *d_a, float *d_b,
                         const int size, const int bufsize);

}; // Operator
```

现在，让我们编写一些顺序 GPU 执行代码，这些代码在前一节中已经使用过，但作为`Operator`类的成员函数，如下所示：

```cpp
void Operator::async_operation(float *h_c, const float *h_a, 
                          const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&_p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, 
                    cudaMemcpyHostToDevice, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, 
                     stream >>>(d_c, d_a, d_b);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, 
                    cudaMemcpyDeviceToHost, stream);

    printf("Launched GPU task %d\n", index);
}
```

这个函数的操作与我们之前使用的基本 CUDA 主机编程模式没有什么不同，只是我们使用了给定的`_stream`应用了`cudaMemcpyAsync()`。然后，我们编写`main()`来处理多个操作符实例和页锁定内存：

```cpp
int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);
```

现在，我们将使用`cudaMallocHost()`来分配主机内存，以获得固定内存，并对其进行初始化：

```cpp
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);
```

而且，我们将拥有相同大小的设备内存：

```cpp
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);
```

现在，我们将使用我们使用的类创建一个 CUDA 操作符列表：

```cpp
    Operator *ls_operator = new Operator[num_operator];
```

我们准备执行流水线操作。在开始执行之前，让我们放一个秒表来查看整体执行时间，并查看重叠数据传输的好处，如下所示：

```cpp
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
```

让我们使用循环执行每个操作符，并且每个操作符将根据其顺序访问主机和设备内存。我们还将测量循环的执行时间：

```cpp
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], 
                                       &h_a[offset], &h_b[offset],
                                       &d_c[offset], 
                                       &d_a[offset], &d_b[offset],
                                       size / num_operator, 
                                       bufsize / num_operator);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
```

最后，我们将比较一个样本的结果，并打印出整体测量性能：

```cpp
    // prints out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n", h_a[print_idx] + 
           h_b[print_idx], h_c[print_idx]);

    // prints out the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / 
                      elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", 
           elapsed_time_msed, bandwidth);
```

终止句柄和内存，如下所示：

```cpp
    sdkDeleteTimer(&timer);
    delete [] ls_operator;
    cudaFree(d_a);    cudaFree(d_b);    cudaFree(d_c);
    cudaFreeHost(h_a);cudaFreeHost(h_b);cudaFreeHost(h_c);
```

要执行代码，让我们重用前面的主机初始化函数和 GPU 内核函数。我们暂时不需要修改这些函数。使用以下命令编译代码：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_pipelining ./cuda_pipelining.cu
```

您必须使用 GPU 的计算能力版本号作为`gencode`选项。编译的输出如下：

```cpp
Launched GPU task 0
Launched GPU task 1
Launched GPU task 2
Launched GPU task 3
compared a sample result...
host: 1.523750, device: 1.523750
Time= 29.508 msec, bandwidth= 27.291121 GB/s
```

正如我们所看到的，GPU 任务是按照内核执行的顺序以及流的顺序执行的。

现在，让我们来回顾一下应用程序在内部是如何运行的。默认情况下，示例代码将主机数据切片为四个部分，并同时执行四个 CUDA 流。我们可以看到每个核函数的输出以及流的执行情况。要查看重叠操作，您需要使用以下命令对执行进行分析：

```cpp
$ nvprof -o overlapping_exec.nvvp ./overlapping_exec
```

以下截图显示了通过重叠数据传输和核函数执行来操作四个 CUDA 流：

![](img/1dc75db9-3e38-4f1f-8309-e4942488014e.png)

核函数执行和数据传输之间的重叠

因此，GPU 可以忙碌直到最后一个核函数执行完成，并且我们可以隐藏大部分数据传输。这不仅增强了 GPU 的利用率，还减少了总应用程序执行时间。

在核函数执行之间，我们可以发现它们虽然属于不同的 CUDA 流，但没有争用。这是因为 GPU 调度器知道执行请求，并优先服务第一个。然而，当当前任务完成时，流多处理器可以为另一个 CUDA 流中的下一个核函数提供服务，因为它们仍然保持占用。

在多个 CUDA 流操作结束时，我们需要同步主机和 GPU，以确认 GPU 上的所有 CUDA 操作都已完成。为此，我们在循环结束后立即使用了`cudaDeviceSynchronize()`。此函数可以在调用点同步所选的所有 GPU 操作。

对于同步任务，我们可以用以下代码替换`cudaDeviceSynchronize()`函数。为此，我们还必须将私有成员`_stream`更改为公共成员：

```cpp
for (int i = 0; i < num_operator; i++) {
    cudaStreamSynchronize(ls_operator[i]._stream);
}
```

当我们需要在每个流完成后从单个主机线程提供特定操作时，可以使用这个。但是，这不是一个好的操作设计，因为后续操作无法避免与其他流同步。

在循环中使用`cudaStreamSynchronize()`怎么样？在这种情况下，我们无法执行之前的重叠操作。以下截图显示了这种情况：

![](img/7c3c1e57-e8fb-4393-9809-7e27ea4e2afc.png)

这是因为`cudaStreamSynchronize()`将同步每次迭代，应用程序将按顺序执行所有 CUDA 执行。在这种情况下，执行时间为 41.521 毫秒，比重叠执行时间慢了约 40%。

# CUDA 回调函数

**CUDA 回调函数**是可调用的主机函数，由 GPU 执行上下文执行。使用此函数，程序员可以指定在 GPU 操作之后执行主机所需的主机操作。

CUDA 回调函数具有一个名为`CUDART_CB`的特殊数据类型，因此应该使用这种类型进行定义。使用此类型，程序员可以指定哪个 CUDA 流启动此函数，传递 GPU 错误状态，并提供用户数据。

要注册回调函数，CUDA 提供了`cudaStreamAddCallback()`。该函数接受 CUDA 流、CUDA 回调函数及其参数，以便从指定的 CUDA 流中调用指定的 CUDA 回调函数并获取用户数据。该函数有四个输入参数，但最后一个是保留的。因此，我们不使用该参数，它保持为`0`。

现在，让我们改进我们的代码，使用回调函数并输出单个流的性能。如果要分开之前的工作和这个工作，可以复制源代码。

首先，将这些函数声明放入`Operator`类的`private`区域：

```cpp
StopWatchInterface *_p_timer;
static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
void print_time();
```

`Callback()`函数将在每个流的操作完成后被调用，`print_time()`函数将使用主机端计时器`_p_timer`报告估计的性能。函数的实现如下：

```cpp
void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData) {
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time() {
    sdkStopTimer(&p_timer);    // end timer
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", index, 
           elapsed_time_msed);
}
```

为了进行正确的计时操作，我们需要在`Operator`类的构造函数中进行计时器初始化，并在类的终结器中进行计时器销毁。此外，我们必须在`Operator::async_operation()`函数的开头启动计时器。然后，在函数的末尾插入以下代码块。这允许 CUDA 流在完成先前的 CUDA 操作时调用主机端函数：

```cpp
// register callback function
cudaStreamAddCallback(stream, Operator::Callback, this, 0);
```

现在，让我们编译并查看执行结果。您必须使用您的 GPU 的计算能力版本号作为`gencode`选项：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_callback ./cuda_callback.cu
```

这是我们更新的执行结果：

```cpp
stream 0 - elapsed 11.136 ms
stream 1 - elapsed 16.998 ms
stream 2 - elapsed 23.283 ms
stream 3 - elapsed 29.487 ms
compared a sample result...
host: 1.523750, device: 1.523750
Time= 29.771 msec, bandwidth= 27.050028 GB/s
```

在这里，我们可以看到估计的执行时间以及 CUDA 流。回调函数估计其序列的执行时间。由于与其他流重叠并延迟后续 CUDA 流，我们可以看到后续 CUDA 流的执行时间延长。我们可以通过与分析结果匹配来确认这些经过的时间，如下所示：

![](img/9cf56a50-a62a-4ccc-839f-a86dbf6526c1.png)

尽管它们的测量经过时间随着流的执行而延长，但流之间的差值是固定的，我们可以从分析输出中看到这些操作。

因此，我们可以得出结论，我们可以编写主机代码，以便在每个单独的 CUDA 流操作完成后立即执行。这比从主线程同步每个流更加先进。

# 具有优先级的 CUDA 流

默认情况下，所有 CUDA 流具有相同的优先级，因此它们可以按正确的顺序执行其操作。此外，CUDA 流还可以具有优先级，并且可以被优先级更高的流取代。有了这个特性，我们可以有满足时间关键要求的 GPU 操作。

# CUDA 中的优先级

要使用具有优先级的流，我们首先需要从 GPU 获取可用的优先级。我们可以使用`cudaDeviceGetStreamPriorityRange()`函数来获取这些值。它的输出是两个数值，即最低和最高的优先级值。然后，我们可以使用`cudaStreamCreaetWithPriority()`函数创建一个优先级流，如下所示：

```cpp
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)
```

我们应该提供两个额外的参数。第一个确定了创建的流与默认流的行为。我们可以使用`cudaStreamDefault`使新流与默认流同步，就像普通流一样。另一方面，我们可以使用`cudaStreamNonBlocking`使其与默认流并行操作。最后，我们可以在优先级范围内设置流的优先级。在 CUDA 编程中，最低值具有最高优先级。

此外，我们可以使用以下代码确认 GPU 是否支持这一点。但是，我们不必太担心这一点，因为自 CUDA 计算能力 3.5 以来，优先级流一直可用：

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
if (prop.streamPrioritiesSupported == 0) { ... }
```

如果设备属性值为`0`，我们应该停止应用程序，因为 GPU 不支持流优先级。

# 具有优先级的流执行

现在，我们将重用之前带有回调的多流应用程序。在这段代码中，我们可以看到流可以按顺序操作，我们将看到如何使用优先级更改这个顺序。我们将从`Operator`类派生一个类，并且它将处理流的优先级。因此，我们将把成员变量流的保护级别从私有成员更改为受保护的成员。构造函数可以选择性地创建流，因为这可以由派生类完成。更改如下代码所示：

```cpp
... { middle of the class Operator } ...
protected:
    cudaStream_t stream = nullptr;

public:
    Operator(bool create_stream = true) {
        if (create_stream)
            cudaStreamCreate(&stream);
        sdkCreateTimer(&p_timer);
    }
... { middle of the class Operator } ...
```

派生类`Operator_with_priority`将具有一个函数，可以根据给定的优先级手动创建一个 CUDA 流。该类的配置如下：

```cpp
class Operator_with_priority: public Operator {
public:
    Operator_with_priority() : Operator(false) {}

    void set_priority(int priority) {
        cudaStreamCreateWithPriority(&stream, 
            cudaStreamNonBlocking, priority);
    }
};
```

当我们使用类处理每个流的操作时，我们将更新`main()`中的`ls_operator`创建代码，以使用我们之前编写的`Operator_with_priority`类，如下所示：

```cpp
Operator_with_priority *ls_operator = new Operator_with_priority[num_operator];
```

当我们更新类时，这个类在我们请求它之前不会创建流。正如我们之前讨论的，我们需要使用以下代码获取 GPU 可用优先级范围：

```cpp
// Get priority range
int priority_low, priority_high;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
printf("Priority Range: low(%d), high(%d)\n", priority_low, priority_high);
```

然后，让我们创建每个操作以拥有不同的优先级流。为了简化这个任务，我们将让最后一个操作拥有最高的流，并看看 CUDA 流中的抢占是如何工作的。可以使用以下代码来实现这一点：

```cpp
for (int i = 0; i < num_operator; i++) {
    ls_operator[i].set_index(i);

    // let the latest CUDA stream to have the high priority
    if (i + 1 == num_operator)
        ls_operator[i].set_priority(priority_high);
    else
        ls_operator[i].set_priority(priority_low);
}
```

之后，我们将执行每个操作，就像之前一样：

```cpp
for (int i = 0 ; i < num_operator; i++) { 
    int offset = i * size / num_operator;
    ls_operator[i].async_operation(&h_c[offset], 
                                   &h_a[offset], &h_b[offset],
                                   &d_c[offset], 
                                   &d_a[offset], &d_b[offset],
                                   size / num_operator, 
                                   bufsize / num_operator);
}
```

为了获得正确的输出，让我们使用`cudaDeviceSynchronize()`函数同步主机和 GPU。最后，我们可以终止 CUDA 流。具有优先级的流可以使用`cudaStreamDestroy()`函数终止，因此在这个应用程序中我们已经做了必要的事情。

现在，让我们编译代码并查看效果。和往常一样，您需要向编译器提供正确的 GPU 计算能力版本：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o prioritized_cuda_stream ./prioritized_cuda_stream.cu
```

接下来是应用程序的输出：

```cpp
Priority Range: low(0), high(-1)
stream 0 - elapsed 11.119 ms
stream 3 - elapsed 19.126 ms
stream 1 - elapsed 23.327 ms
stream 2 - elapsed 29.422 ms
compared a sample result...
host: 1.523750, device: 1.523750
Time= 29.730 msec, bandwidth= 27.087332 GB/s
```

从输出中，您可以看到操作顺序已经改变。Stream 3 在 Stream 1 和 Stream 2 之前。下面的屏幕截图显示了它是如何改变的：

![](img/84f4080c-7d72-4b4c-ad7c-b0192fe69d97.png)

在这个屏幕截图中，第二个 CUDA 流（在这种情况下是 Stream 19）被优先级最低的 CUDA 流（Stream 21）抢占，以便在 Stream 21 执行完毕后 Stream 19 完成其工作。请注意，数据传输的顺序不会根据这种优先级而改变。

# 使用 CUDA 事件估计内核执行时间

以前的 GPU 操作时间估计有一个限制，即它无法测量内核执行时间。这是因为我们在主机端使用了计时 API。因此，我们需要与主机和 GPU 同步以测量内核执行时间，考虑到对应用程序性能的开销和影响，这是不切实际的。

这可以通过使用 CUDA 事件来解决。CUDA 事件记录 GPU 端的事件以及 CUDA 流。CUDA 事件可以是基于 GPU 状态的事件，并记录调度时间。使用这个，我们可以触发以下操作或估计内核执行时间。在本节中，我们将讨论如何使用 CUDA 事件测量内核执行时间。

CUDA 事件由`cudaEvent_t`句柄管理。我们可以使用`cudaEventCreate()`创建 CUDA 事件句柄，并使用`cudaEventDestroy()`终止它。要记录事件时间，可以使用`cudaEventRecord()`。然后，CUDA 事件句柄记录 GPU 的事件时间。这个函数还接受 CUDA 流，这样我们就可以将事件时间枚举到特定的 CUDA 流。在获取内核执行的开始和结束事件之后，可以使用`cudaEventElapsedTime()`获取经过的时间，单位为毫秒。

现在，让我们讨论如何使用 CUDA 事件来使用这些 API。

# 使用 CUDA 事件

在本节中，我们将重用第二节中的多流应用程序。然后，我们使用 CUDA 事件枚举每个 GPU 内核的执行时间：

1.  我们将使用一个简单的向量加法内核函数，如下所示：

```cpp
__global__ void
vecAdd_kernel(float *c, const float* a, const float* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}
```

这段代码有一个迭代，它延长了内核执行时间。

1.  然后，我们将使用以下片段来测量内核执行时间。为了比较结果，我们将使用主机端的计时器和 CUDA 事件：

```cpp
... { memory initializations } ...

// initialize the host timer
StopWatchInterface *timer;
sdkCreateTimer(&timer);

cudaEvent_t start, stop;
// create CUDA events
cudaEventCreate(&start);
cudaEventCreate(&stop);

// start to measure the execution time
sdkStartTimer(&timer);
cudaEventRecord(start);

// launch cuda kernel
dim3 dimBlock(256);
dim3 dimGrid(size / dimBlock.x);
vecAdd_kernel<<< dimGrid, dimBlock >>>(d_c, d_a, d_b);

// record the event right after the kernel execution finished
cudaEventRecord(stop);

// Synchronize the device to measure the execution time from the host side
cudaEventSynchronize(stop); // we also can make synchronization based on CUDA event
sdkStopTimer(&timer);
```

正如您在这段代码中所看到的，我们可以在内核调用之后立即记录 CUDA 事件。然而，计时器需要在 GPU 和主机之间进行同步。为了同步，我们使用`cudaEventSynchronize(stop)`函数，因为我们也可以使主机线程与事件同步。与此同时，这段代码只涵盖了处理计时资源和内核执行。但是，您还需要初始化所需的内存才能使其工作。

1.  在内核执行之后，让我们编写代码报告每个计时资源的执行时间：

```cpp
// print out the result
int print_idx = 256;
printf("compared a sample result...\n");
printf("host: %.6f, device: %.6f\n", h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

// print estimated kernel execution time
float elapsed_time_msed = 0.f;
cudaEventElapsedTime(&elapsed_time_msed, start, stop);
printf("CUDA event estimated - elapsed %.3f ms \n", elapsed_time_msed);
```

1.  现在，我们将通过终止计时资源来完成我们的应用程序，使用以下代码：

```cpp
// delete timer
sdkDeleteTimer(&timer);

// terminate CUDA events
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

1.  让我们编译并使用以下命令查看输出：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_event ./cuda_event.cu
compared a sample result...
host: 1.523750, device: 1.523750
CUDA event estimated - elapsed 23.408 ms 
Host measured time= 35.063 msec/s
```

如您所见，我们可以使用 CUDA 事件来测量内核执行时间。但是，测量的时间在 CUDA 事件和计时器之间存在间隙。我们可以使用 NVIDIA 分析器来验证哪个提供更准确的信息。当我们使用`# nvprof ./cuda_event`命令时，输出如下：

![](img/533f3b7e-846e-4f23-a97d-b277876a05c4.png)

如您所见，与从主机测量相比，CUDA 事件提供了准确的结果。

使用 CUDA 事件的另一个好处是，我们可以使用多个 CUDA 流同时测量多个内核执行时间。让我们实现一个示例应用程序并查看其操作。

# 多流估计

`cudaEventRecord()`函数对主机是异步的。换句话说，没有同步来测量内核执行时间到示例代码。为了使事件和主机同步，我们需要使用`cudaEventSynchronize()`。例如，当我们在`cudaEventRecord(stop)`之后立即放置这个函数时，可以在设备到主机的异步数据传输之前放置内核函数打印，通过同步效果来实现。

在多个 CUDA 流应用程序中测量内核执行时间也是有用的：

1.  让我们将这应用到`04_stream_priority`示例代码中的多个 CUDA 流重叠的代码中。使用以下代码更新代码：

```cpp
class Operator
{
private:
    int _index;
    cudaStream_t stream;
    StopWatchInterface *p_timer;
    cudaEvent_t start, stop;

public:
    Operator() {
        cudaStreamCreate(&stream);

 // create cuda event
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
    }

    ~Operator() {
        cudaStreamDestroy(stream);

 // destroy cuda event
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
    }

    void set_index(int idx) { index = idx; }
    void async_operation(float *h_c, const float *h_a, 
                          const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);
 void print_kernel_time();

}; // Operator
```

1.  然后，我们将定义此时包含的`print_time()`函数，如下所示：

```cpp
void Operator::print_time() {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Stream %d time: %.4f ms\n", index, milliseconds);
}
```

1.  现在，在`Operator::async_operation()`的开头和结尾插入`cudaEventRecord()`函数调用，如下所示：

```cpp
void Operator::async_operation( ... )
{
    // start timer
    sdkStartTimer(&p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, 
                    cudaMemcpyHostToDevice, stream);

    // record the event before the kernel execution
 cudaEventRecord(start, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, 
                     stream >>>(d_c, d_a, d_b);

    // record the event right after the kernel execution finished
 cudaEventRecord(stop, stream);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, 
                    cudaMemcpyDeviceToHost, stream);

    // what happen if we include CUDA event synchronize?
    // QUIZ: cudaEventSynchronize(stop);

    // register callback function
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}
```

对于这个函数，在函数的末尾放置同步是一个挑战。在完成本节后尝试这样做。这将影响应用程序的行为。建议尝试自己解释输出，然后使用分析器进行确认。

现在，让我们编译并查看执行时间报告，如下；它显示与先前执行类似的性能：

```cpp
$ nvcc -m64 -run -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -o cuda_event_with_streams ./cuda_event_with_streams.cu
Priority Range: low(0), high(-1)
stream 0 - elapsed 11.348 ms 
stream 3 - elapsed 19.435 ms 
stream 1 - elapsed 22.707 ms 
stream 2 - elapsed 35.768 ms 
kernel in stream 0 - elapsed 6.052 ms 
kernel in stream 1 - elapsed 14.820 ms 
kernel in stream 2 - elapsed 17.461 ms 
kernel in stream 3 - elapsed 6.190 ms 
compared a sample result...
host: 1.523750, device: 1.523750
Time= 35.993 msec, bandwidth= 22.373972 GB/s
```

在这个输出中，我们还可以看到每个内核的执行时间，这要归功于 CUDA 事件。从这个结果中，我们可以看到内核执行时间延长了，就像我们在上一节中看到的那样。

如果您想了解更多关于 CUDA 事件特性的信息，请查看 NVIDIA 的 CUDA 事件文档：[`docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)。

现在，我们将介绍管理 CUDA 网格的其他一些方面。第一项是动态并行性，它使 GPU 内核函数能够进行内核调用。

# CUDA 动态并行性

**CUDA 动态并行性**（**CDP**）是一种设备运行时功能，它允许从设备函数进行嵌套调用。这些嵌套调用允许子网格具有不同的并行性。当问题需要不同的块大小时，此功能非常有用。

# 理解动态并行性

与主机的普通内核调用一样，GPU 内核调用也可以进行内核调用。以下示例代码显示了它的工作原理：

```cpp
__global__ void child_kernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&data[idx], seed);
}

__global__ void parent_kernel(int *data)
{
 if (threadIdx.x == 0) {
        int child_size = BUF_SIZE/gridDim.x;
        child_kernel<<< child_size/BLOCKDIM, BLOCKDIM >>>
                        (&data[child_size*blockIdx.x], blockIdx.x+1);
    }
    // synchronization for other parent's kernel output
    cudaDeviceSynchronize();
}
```

如您在这些函数中所见，我们需要确保哪个 CUDA 线程进行内核调用以控制网格创建的数量。要了解更多信息，让我们使用这个实现第一个应用程序。

# 动态并行性的使用

我们的动态并行性代码将创建一个父网格，该父网格将创建一些子网格：

1.  首先，我们将使用以下代码编写`parent_kernel()`函数和`child_kernel()`函数：

```cpp
#define BUF_SIZE (1 << 10)
#define BLOCKDIM 256

__global__ void child_kernel(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&data[idx], 1);
}

__global__ void parent_kernel(int *data)
{
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    {
        int child_size = BUF_SIZE/gridDim.x;
        child_kernel<<< child_size/BLOCKDIM, BLOCKDIM >>> \
                        (&data[child_size*blockIdx.x], 
                         blockIdx.x+1);
    }
    // synchronization for other parent's kernel output
    cudaDeviceSynchronize();
}
```

如您在这段代码中所见，父内核函数创建子内核网格作为块的数量。然后，子网格递增指定的内存`1`来标记它们的操作。内核执行后，父内核使用`cudaDeviceSynchronize()`函数等待所有子网格完成其工作。在进行同步时，我们应确定同步的范围。如果我们需要在块级别进行同步，我们应选择`__synchthread()`。

1.  使用以下代码编写`main()`函数：

```cpp
#define BUF_SIZE (1 << 10)
#define BLOCKDIM 256
int main()
{
    int *data;
    int num_child = 4;

    cudaMallocManaged((void**)&data, BUF_SIZE * sizeof(int));
    cudaMemset(data, 0, BUF_SIZE * sizeof(int));

    parent_kernel<<<num_child, 1>>>(data);
    cudaDeviceSynchronize();

    // Count elements value
    int counter = 0;
    for (int i = 0; i < BUF_SIZE; i++)
        counter += data[i];

    // getting answer
    int counter_h = 0;
    for (int i = 0; i < num_child; i++)
        counter_h += (i+1);
    counter_h *= BUF_SIZE / num_child;

    if (counter_h == counter)
        printf("Correct!!\n");
    else
        printf("Error!! Obtained %d. It should be %d\n", 
               counter, counter_h);

    cudaFree(data);
    return 0;
}
```

正如前面讨论的，我们将创建子网格以及块的数量。因此，我们将使用网格大小为`4`来执行父内核函数，而块大小为`1`。

1.  要编译 CDP 应用程序，我们应该为`nvcc`编译器提供`-rdc=true`选项。因此，编译源代码的命令如下：

```cpp
$ nvcc -run -rdc=true -lcudadevrt -gencode arch=compute_70,code=sm_70 -o host_callback host_callback.cu -I/usr/local/cuda/samples/common/inc 
```

1.  让我们对这个应用程序进行分析，以了解其操作。以下截图显示了这个嵌套调用的工作原理：

![](img/765aef65-f755-4eb1-9697-ff07d3efb6d7.png)

如我们在这个屏幕截图中所见，父内核创建了一个子网格，我们可以在左侧面板的右角标中看到它们的关系。然后，父网格（parent_kernel）等待其执行，直到子网格完成其工作。CUDA 目前不支持 SM70（Volta 架构）的 CDT 分析，因此我使用 Tesla P40 来获得这个输出。

# 递归

动态并行性的一个好处是我们可以创建递归。以下代码显示了一个递归内核函数的示例：

```cpp
__global__ void recursive_kernel(int *data, int size, int depth) {
  int x_0 = blockIdx.x * size;

  if (depth > 0) {
    __syncthreads();
 if (threadIdx.x == 0) {
        int dimGrid = size / dimBlock;
        recursive_kernel<<<dimGrid, 
              dimBlock>>>(&data[x_0], size/dimGrid, depth-1);
        cudaDeviceSynchronize();
      }
      __syncthreads();
   }
}
```

如您所见，与以前的动态并行内核函数相比，没有太大的区别。但是，我们应该谨慎使用这个功能，考虑到资源使用和限制。一般来说，动态并行内核可以保守地保留高达 150MB 的设备内存来跟踪待处理的网格启动和通过在子网格启动上进行同步来同步父网格的状态。此外，同步必须在多个级别上小心进行，而嵌套内核启动的深度限制为 24 级。最后，控制嵌套内核启动的运行时可能会影响整体性能。

如果您需要了解动态并行性的限制和限制，请参阅以下编程指南：[`docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implementation-restrictions-and-limitations`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implementation-restrictions-and-limitations)。

我们将在第七章中介绍其在 CUDA 中快速排序实现中的应用，即*CUDA 中的并行编程模式*。要了解更多关于动态并行性的信息，请参阅以下文档：

+   https://devblogs.nvidia.com/cuda-dynamic-parallelism-api-principles/

+   [`on-demand.gputechconf.com/gtc/2012/presentations/S0338-New-Features-in-the-CUDA-Programming-Model.pdf`](http://on-demand.gputechconf.com/gtc/2012/presentations/S0338-New-Features-in-the-CUDA-Programming-Model.pdf)

# 网格级别的协作组

如 第三章 中所讨论的，CUDA 提供了协作组。协作组可以根据其分组目标进行分类：warp 级别、块级别和网格级别的组。本文介绍了网格级别的协作组，并探讨了协作组如何处理 CUDA 网格。

协作组最显著的好处是对目标并行对象的显式同步。使用协作组，程序员可以设计他们的应用程序来显式同步 CUDA 并行对象、线程块或网格。使用第三章中介绍的块级协作组，*CUDA 线程编程*，我们可以通过指定需要同步的 CUDA 线程或块来编写更易读的代码。

# 理解网格级协作组

自 9.0 版本以来，CUDA 提供了另一级协作组，与网格一起工作。具体来说，有两个网格级协作组：`grid_group`和`multi_grid_group`。使用这些组，程序员可以描述网格在单个 GPU 或多个 GPU 上的操作同步。

在这个示例中，我们将探索`grid_group`的功能，它可以同步网格与减少问题，就像第三章中所提到的，*CUDA 线程编程*，关于基于块级减少的先前减少设计。每个线程块产生自己的减少结果，并将它们存储到全局内存中。然后，另一个块级减少内核启动，直到我们获得单个减少值。这是因为完成内核操作可以保证下一个**减少**内核从多个线程块中读取减少值。其设计由左侧的图表描述：

！[](img/e70bce0c-8537-4ad1-a5c6-82cc22c93cb0.png)

另一方面，网格级同步使另一种内部同步块式**减少**结果的内核设计成为可能，以便主机只需调用一次内核即可获得减少**结果**。在协作组中，`grid_group.sync()`提供了这样的功能，因此我们可以编写减少内核而无需内核级迭代。

要使用`grid_group.sync()`函数，我们需要使用`cudaLaunchCooperativeKernel()`函数调用内核函数。其接口设计如下：

```cpp
__host__ cudaError_t cudaLaunchCooperativeKernel
    ( const T* func, dim3 gridDim, dim3 blockDim, 
      void** args, size_t sharedMem = 0, cudaStream_t stream = 0 )
```

因此，它的使用方式与`cudaLaunchKernel()`函数相同，该函数启动内核函数。

为了使`grid_group`中的所有线程块同步，网格中活动线程块的总数不应超过内核函数和设备的最大活动块数。GPU 上的最大活动块大小是每个 SM 的最大活动块数和流处理器的数量的乘积。违反此规则可能导致死锁或未定义行为。我们可以使用`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`函数来获取每个 SM 内核函数的最大活动线程块数，通过传递内核函数和块大小信息。

# 使用`grid_group`的用法

现在，让我们将`grid_group`应用于并行减少问题，并看看 GPU 编程如何改变：

1.  我们将重用之前并行减少代码中的主机代码，即`03_cuda_thread_programming/07_cooperative_groups`。换句话说，我们将通过对主机代码进行小的更改来改变 GPU 的操作。您还可以使用`07_grid_level_cg`目录中的代码。

1.  现在，让我们编写一些块级减少代码。当我们有网格级协作组时，所有线程块必须是活动的。换句话说，我们不能执行多个线程块，而 GPU 能够执行的活动块。因此，这个减少将首先累积输入数据，以覆盖所有数据，使用有限数量的线程块。然后，它将在块级进行并行减少，就像我们在第三章中所介绍的那样，*CUDA 线程编程*。

以下代码显示了它的实现：

```cpp
__device__ void
block_reduction(float *out, float *in, float *s_data, int active_size, int size, 
          const cg::grid_group &grid, const cg::thread_block &block)
{
  int tid = block.thread_rank();

  // Stride over grid and add the values to a shared memory buffer
  s_data[tid] = 0.f;
  for (int i = grid.thread_rank(); i < size; i += active_size)
    s_data[tid] += in[i];

  block.sync();

  for (unsigned int stride = blockDim.x / 2; 
       stride > 0; stride >>= 1) {
    if (tid < stride)
      s_data[tid] += s_data[tid + stride];
    block.sync();
  }

  if (block.thread_rank() == 0)
    out[block.group_index().x] = s_data[0];
}
```

1.  然后，让我们编写一个内核函数，考虑活动块数和`grid_group`执行块级减少。在这个函数中，我们将调用块级减少代码，并在网格级别进行同步。然后，我们将从输出中执行并行减少，就像我们在第三章 *CUDA 线程编程*中所介绍的那样。以下代码显示了其实现：

```cpp
__global__ void
reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
  cg::thread_block block = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float s_data[];

  // do reduction for multiple blocks
  block_reduction(g_out, g_in, s_data, grid.size(), 
                  size, grid, block);

  grid.sync();

  // do reduction with single block
  if (block.group_index().x == 0)
    block_reduction(g_out, g_out, s_data, block.size(), gridDim.x, grid, block);
}
```

1.  最后，我们将实现调用具有可用活动线程块维度的内核函数的主机代码。为此，此函数使用`cudaoccupancyMaxActiveBlocksPerMultiprocessor()`函数。此外，网格级合作组要求我们通过`cudaLaunchCooperativeKernel()`函数调用内核函数。您可以在这里看到实现：

```cpp
int reduction_grid_sync(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{ 
  int num_blocks_per_sm;
  cudaDeviceProp deviceProp;

  // Calculate the device occupancy to know 
  // how many blocks can be run concurrently
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, 
      reduction_kernel, n_threads, n_threads*sizeof(float));
  int num_sms = deviceProp.multiProcessorCount;
  int n_blocks = min(num_blocks_per_sm * num_sms, 
                     (size + n_threads - 1) / n_threads);

  void *params[3];
  params[0] = (void*)&g_outPtr;
  params[1] = (void*)&g_inPtr;
  params[2] = (void*)&size;
  cudaLaunchCooperativeKernel((void*)reduction_kernel, 
                              n_blocks, n_threads, params, 
                              n_threads * sizeof(float), NULL);

  return n_blocks;
}
```

1.  现在，请确保可以从`reduction.cpp`文件中调用主机函数。

1.  然后，让我们编译代码并查看其操作。以下 shell 命令编译代码并执行应用程序。计算能力应该等于或大于`70`：

```cpp
$ nvcc -run -m64 -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -rdc=true -o reduction ./reduction.cpp ./reduction_kernel.cu
Time= 0.474 msec, bandwidth= 141.541077 GB/s
host: 0.996007, device 0.996007
```

输出性能远远落后于我们在第三章 *CUDA 线程编程*的最终结果。由于`block_reduction()`函数在开始时使用了高内存吞吐量，因此它是高度内存绑定的：

![](img/fbd18a08-d2a1-4bc1-a9b3-647c67a72240.png)

主要影响因素是我们只能使用活动线程块。因此，我们无法隐藏内存访问时间。实际上，使用`grid_group`还有其他目的，例如图搜索、遗传算法和粒子模拟，这要求我们保持状态长时间处于活动状态以获得性能。

这种网格级同步可以为性能和可编程性提供更多好处。由于这使得内核可以自行同步，我们可以使内核自行迭代。因此，它对解决图搜索、遗传算法和实际模拟非常有用。要了解有关`grid_groups`中合作组的更多信息，请参阅提供的文档[`on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf`](http://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)。

# 使用 OpenMP 的 CUDA 内核调用

为了增加应用程序的并发性，我们可以从主机的并行任务中进行内核调用。例如，OpenMP 提供了多核架构的简单并行性。本教程介绍了 CUDA 如何操作 OpenMP。

# OpenMP 和 CUDA 调用

OpenMP 使用分叉-合并模型的并行性来针对多核 CPU。主线程启动并行操作并创建工作线程。主机线程并行运行自己的工作，并在完成工作后加入。

使用 OpenMP，CUDA 内核调用可以与多个线程并行执行。这有助于程序员不必维护单独的内核调用，而是允许它们的内核执行依赖于主机线程的索引。

在本节中，我们将使用以下 OpenMP API：

+   `omp_set_num_threads()`设置将并行工作的工作线程数。

+   `omp_get_thread_num()`返回工作线程的索引，以便每个线程可以识别其任务。

+   `#pragma omp parallel {}` 指定了一个并行区域，将由工作线程覆盖。

现在，让我们编写一些代码，其中 OpenMP 调用 CUDA 内核函数。

# CUDA 与 OpenMP 的内核调用

在本节中，我们将实现一个使用 OpenMP 的多流矢量加法应用程序。为此，我们将修改先前的版本并查看差异：

1.  要测试 CUDA 中的 OpenMP，我们将修改`03_cuda_callback`目录中的代码。我们将修改`main()`函数的主体，或者您可以使用放置在`08_openmp_cuda`目录中的提供的示例代码。

1.  现在，让我们包括 OpenMP 头文件并修改代码。要在代码中使用 OpenMP，我们应该使用`#include <omp.h>`。而且，我们将更新代码，使其使用 OpenMP 来迭代每个流：

```cpp
// execute each operator collesponding data
omp_set_num_threads(num_operator);
#pragma omp parallel
{
    int i = omp_get_thread_num();
    printf("Launched GPU task %d\n", i);

    int offset = i * size / num_operator;
    ls_operator[i].set_index(i);
    ls_operator[i].async_operation(&h_c[offset], &h_a[offset],   
                                   &h_b[offset],&d_c[offset], 
                                   &d_a[offset], &d_b[offset],
                                   size / num_operator, bufsize 
                                   / num_operator);
}
```

1.  使用以下命令编译代码：

```cpp
$ nvcc -run -m64 -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -Xcompiler -fopenmp -lgomp -o openmp ./openmp.cu
stream 0 - elapsed 10.734 ms 
stream 2 - elapsed 16.153 ms 
stream 3 - elapsed 21.968 ms 
stream 1 - elapsed 27.668 ms 
compared a sample result...
host: 1.523750, device: 1.523750
Time= 27.836 msec, bandwidth= 28.930389 GB/s
```

每当您执行此应用程序时，您将看到每个流以无序方式完成其工作。此外，每个流显示不同的时间。这是因为 OpenMP 可以创建多个线程，并且操作是在运行时确定的。

为了了解其运行情况，让我们对应用程序进行分析。以下截图显示了应用程序的分析时间表。由于调度的原因，这可能与您的情况不同：

![](img/50012166-e450-4d38-8ebf-a44a55ad9844.png)

如您在此截图中所见，您将能够看到数据传输与 Stream 17 相比已经反转。因此，我们可以看到第二个流最终完成了它的工作。

# 多进程服务

GPU 能够从并发的 CPU 进程中执行内核。但是，默认情况下，它们只以分时方式执行，即使每个内核没有充分利用 GPU 计算资源。为了解决这种不必要的串行化，GPU 提供了**多进程服务**（**MPS**）模式。这使得不同的进程能够同时在 GPU 上执行它们的内核，以充分利用 GPU 资源。启用时，`nvidia-cuda-mps-control`守护进程监视目标 GPU，并使用该 GPU 管理进程内核操作。此功能仅在 Linux 上可用。在这里，我们可以看到多个进程共享同一个 GPU 的 MPS：

![](img/2f97d170-6740-41d4-aa8b-7552745e9838.jpg)

正如我们所看到的，每个进程在 GPU 上并行运行一部分（绿色条），而一部分在 CPU 上运行（蓝色条）。理想情况下，您需要蓝色条和绿色条都能获得最佳性能。这可以通过利用所有最新 GPU 支持的 MPS 功能来实现。

请注意，当一个 MPI 进程无法饱和整个 GPU 并且代码的重要部分也在 CPU 上运行时，多个 MPI 进程在同一个 GPU 上运行是有益的。如果一个 MPI 进程利用整个 GPU，即使 CPU 部分（蓝色条）会减少，绿色条的时间也不会减少，因为 GPU 完全被一个 MPI 进程利用。其他 MPI 进程将根据 GPU 架构以分时方式依次访问 GPU。这类似于启动并发内核的情况。如果一个内核利用整个 GPU，那么另一个内核要么等待第一个内核完成，要么进行分时。

这样做的好处是不需要对应用程序进行任何更改即可使用 MPS。MPS 进程作为守护进程运行，如下命令所示：

```cpp
$nvidia-smi -c EXCLUSIVE_PROCESS 
$nvidia-cuda-mps-control –d
```

运行此命令后，所有进程都将其命令提交给 MPS 守护进程，该守护进程负责将 CUDA 命令提交给 GPU。对于 GPU，只有一个进程访问 GPU（MPS 守护进程），因此多个进程可以同时运行来自多个进程的多个内核。这可以帮助将一个进程的内存复制与其他 MPI 进程的内核执行重叠。

# 消息传递接口简介

**消息传递接口**（**MPI**）是一种并行计算接口，它能够触发多个进程跨计算单元 - CPU 核心、GPU 和节点。典型的密集多 GPU 系统包含 4-16 个 GPU，而 CPU 核心的数量在 20-40 个之间。在启用 MPI 的代码中，应用程序的某些部分作为不同的 MPI 进程在多个核心上并行运行。每个 MPI 进程都将调用 CUDA。了解将 MPI 进程映射到相应的 GPU 非常重要。最简单的映射是 1:1，即每个 MPI 进程都独占相应的 GPU。此外，我们还可以将多个 MPI 进程理想地映射到单个 GPU 上。

为了将多进程应用场景应用到单个 GPU 上，我们将使用 MPI。要使用 MPI，您需要为您的系统安装 OpenMPI。按照以下步骤在 Linux 上安装 OpenMPI。此操作已在 Ubuntu 18.04 上进行了测试，因此如果您使用其他发行版，可能会有所不同：

```cpp
$ wget -O /tmp/openmpi-3.0.4.tar.gz https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.4.tar.gz
$ tar xzf /tmp/openmpi-3.0.4.tar.gz -C /tmp
$ cd /tmp/openmpi-3.0.4
$ ./configure --enable-orterun-prefix-by-default --with-cuda=/usr/local/cuda
$ make -j $(nproc) all && sudo make install
$ sudo ldconfig
$ mpirun --version
mpirun (Open MPI) 3.0.4

Report bugs to http://www.open-mpi.org/community/help/
```

现在，让我们实现一个可以与 MPI 和 CUDA 一起工作的应用程序。

# 实现一个启用 MPI 的应用程序

要使应用程序与 MPI 一起工作，我们需要在应用程序中放入一些可以理解 MPI 命令的代码：

1.  我们将重用 OpenMP 示例代码，因此将`openmp.cu`文件复制到`08_openmp_cuda`目录中。

1.  在代码开头插入`mpi`头文件`include`语句：

```cpp
#include <mpi.h>
```

1.  在`main()`函数中创建秒表后立即插入以下代码：

```cpp
// set num_operator as the number of requested process
int np, rank;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &np);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```

1.  按照第 3 步中提到的代码后，将所需的内存大小切割为进程数，如下所示：

```cpp
bufsize /= np;
size /= np;
```

1.  我们需要让每个线程报告它们所属的进程。让我们更新并行执行代码块中的`printf()`函数，如下所示：

```cpp
// execute each operator collesponding data
omp_set_num_threads(num_operator);
#pragma omp parallel
{
    int i = omp_get_thread_num();
    int offset = i * size / num_operator;
    printf("Launched GPU task (%d, %d)\n", rank, i);

    ls_operator[i].set_index(i);
    ls_operator[i].async_operation(&h_c[offset], 
                                   &h_a[offset], &h_b[offset],
                                   &d_c[offset], &d_a[offset], 
                                   &d_b[offset],
                                   size / num_operator, 
                                   bufsize / num_operator);
}
```

1.  在`main()`的末尾放置`MPI_Finalize()`函数以关闭 MPI 实例。

1.  使用以下命令编译代码：

```cpp
$ nvcc -m64 -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -I/usr/local/include/ -Xcompiler -fopenmp -lgomp -lmpi -o simpleMPI ./simpleMPI.cu
```

您必须使用 GPU 的计算能力版本号来选择`gencode`选项。

1.  使用以下命令测试编译后的应用程序：

```cpp
$ ./simpleMPI 2
```

1.  现在，使用以下命令测试 MPI 执行：

```cpp
$ mpirun -np 2 ./simpleMPI 2
Number of process: 2
Number of operations: 2
Launched GPU task (1, 0)
Launched GPU task (1, 1)
Number of operations: 2
Launched GPU task (0, 0)
Launched GPU task (0, 1)
stream 0 - elapsed 13.390 ms 
stream 1 - elapsed 25.532 ms 
compared a sample result...
host: 1.306925, device: 1.306925
Time= 25.749 msec, bandwidth= 15.637624 GB/s
stream 0 - elapsed 21.334 ms 
stream 1 - elapsed 26.010 ms 
compared a sample result...
host: 1.306925, device: 1.306925
Time= 26.111 msec, bandwidth= 15.420826 GB/s
```

# 启用 MPS

在 GPU 上启用 MPS 需要对 GPU 操作模式进行一些修改。但是，您需要具有比 Kepler 架构更晚的 GPU 架构。

让我们按照以下步骤启用 MPS：

1.  使用以下命令启用 MPS 模式：

```cpp
$ export CUDA_VISIBLE_DEVICES=0
$ sudo nvidia-smi -i 0 -c 3
$ sudo nvidia-cuda-mps-control -d
```

或者，您可以使用`make enable_mps`命令来使用此预定义在`Makefile`中的配方示例代码。然后，我们可以从`nivida-smi`输出中看到更新后的计算模式：

![](img/329cace9-b5f9-43e7-8716-c2aa40c0547f.png)

1.  现在，使用以下命令测试 MPS 模式下的 MPI 执行：

```cpp
$ mpirun -np 2 ./simpleMPI 2
Number of process: 2
Number of operations: 2
Launched GPU task (1, 0)
Launched GPU task (1, 1)
stream 0 - elapsed 10.203 ms 
stream 1 - elapsed 15.903 ms 
compared a sample result...
host: 1.306925, device: 1.306925
Time= 16.129 msec, bandwidth= 24.964548 GB/s
Number of operations: 2
Launched GPU task (0, 0)
Launched GPU task (0, 1)
stream 0 - elapsed 10.203 ms 
stream 1 - elapsed 15.877 ms 
compared a sample result...
host: 1.306925, device: 1.306925
Time= 15.997 msec, bandwidth= 25.170544 GB/s
```

如您所见，与之前的执行相比，每个进程的经过时间都有所减少。

1.  现在，让我们恢复原始模式。要禁用 MPS 模式，请使用以下命令：

```cpp
$ echo "quit" | sudo nvidia-cuda-mps-control
$ sudo nvidia-smi -i 0 -c 0
```

或者，您可以使用`make disable_mps`命令来使用此预定义在`Makefile`中的配方示例代码。

要了解更多关于 MPS 的信息，请使用以下链接：

+   [`on-demand.gputechconf.com/gtc/2015/presentation/S5584-Priyanka-Sah.pdf`](http://on-demand.gputechconf.com/gtc/2015/presentation/S5584-Priyanka-Sah.pdf)

+   [`docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf`](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)

# 对 MPI 应用程序进行分析并了解 MPS 操作

使用 MPI，多个进程的内核可以同时共享 GPU 资源，从而增强整体 GPU 利用率。没有 MPS，由于时间切片共享和上下文切换开销，GPU 资源被低效地共享。

以下屏幕截图显示了没有 MPS 的多个进程的时间轴配置文件结果：

![](img/13ccf0f1-20b5-40be-8738-7b460f2f5cac.png)

在此配置文件中，我们可以看到两个 CUDA 上下文共享一个 GPU，并且由于上下文之间的时间共享，内核执行时间延长。

另一方面，MPS 模式管理内核执行请求，因此所有内核执行都会像使用单个进程一样启动。以下屏幕截图显示了 MPS 模式下的内核执行：

![](img/37c34d58-26dd-41ce-8289-65c68a535761.png)

如您所见，只有一个 CUDA 流驻留在 GPU 上并控制所有 CUDA 流。此外，所有内核执行时间都得到了稳定，并且使用 MPS 可以减少总的经过时间。总之，使用 MPS 模式有利于多个 GPU 进程的整体性能，并共享 GPU 资源。

`nvprof`支持将多个 MPI 进程的分析器信息转储到不同的文件中。例如，对于基于 Open MPI 的应用程序，以下命令将在多个文件中转储分析信息，每个文件的名称都基于 MPI 进程的排名：

```cpp
$ mpirun -np 2 nvprof -f -o simpleMPI.%q{OMPPI_COMM_WORLD_RANK}_2.nvvp ./simpleMPI 2
```

或者，您可以使用以下命令来执行示例代码：

```cpp
$ PROCS=2 STREAMS=2 make nvprof
```

然后，您将为每个进程获得两个`nvvp`文件。

现在，我们将使用以下步骤使用 NVIDIA Visual Profiler 来查看这些`nvvp`文件：

1.  打开文件|导入菜单，通过导入`nvvp`文件创建一个分析会话：

![](img/3f802cd1-25da-4e02-b172-9e41b730799f.png)

在 Windows 或 Linux 中，快捷键是*Ctrl* + *I*，OSX 使用*command* + *I*。

1.  然后从列表中选择 Nvprof 后，点击下一步按钮：

![](img/2f84e5d6-9933-4c8d-81d1-e6a6bd03545e.png)

1.  从 Nvprof 选项中，选择多个进程，然后单击下一步>：

![](img/e70799cb-c6f8-4e70-abb6-f8704a433c83.png)

1.  从导入 Nvprof 数据中，单击浏览...按钮，并选择由`nvprof`生成的`nvvp`文件。要对具有多个进程的应用程序进行分析，您需要导入`nvvp`文件，因为存在多个进程：

![](img/1582fbfc-ddd9-42b4-882e-23772b33e767.png)

1.  单击完成，然后 NVIDIA Visual Profiler 将以时间线视图显示分析结果，如下所示：

![](img/75dd51b7-edb6-4d14-80d3-a73b6725191d.png)

请注意，只有同步 MPI 调用将由`nvprof`进行注释。如果使用异步 MPI API，则需要使用其他 MPI 专用的分析工具。其中一些最著名的工具包括以下内容：

+   **TAU**：TAU 是一种性能分析工具包，目前由俄勒冈大学维护。

+   **Vampir**：这是一种商业可用的工具，对数百个 MPI 进程具有良好的可伸缩性。

+   **Intel VTune Amplifier**：商业工具的另一个选择是 Intel VTune Amplifier。它是目前可用的最好的工具之一，可用于 MPI 应用程序分析。

最新的 CUDA 工具包还允许对 MPI API 进行注释。为此，需要将`--annotate-mpi`标志传递给`nvprof`，如以下命令所示：

```cpp
mpirun -np 2 nvprof --annotate-mpi openmpi -o myMPIApp.%q{OMPI_COMM_WORLD_RANK}.nvprof ./myMPIApplciation
```

# 内核执行开销比较

对于迭代并行 GPU 任务，我们有三种内核执行方法：迭代内核调用，具有内部循环，以及使用动态并行性进行递归。最佳操作由算法和应用程序确定。但是，您也可以考虑它们之间的内核执行选项。本示例帮助您比较这些内核执行开销并审查它们的可编程性。

首先，让我们确定我们将测试哪种操作。本示例将使用一个简单的 SAXPY 操作。这有助于我们专注并制作迭代执行代码。此外，随着操作变得更简单，操作控制开销将变得更重。但是，您当然可以尝试任何其他操作。

# 实现三种内核执行方式

以下步骤涵盖了三种不同迭代操作的性能比较：

1.  创建并导航到`10_kernel_execution_overhead`目录。

1.  编写`simple_saxpy_kernel()`函数，代码如下：

```cpp
__global__ void
simple_saxpy_kernel(float *y, const float* x, const float alpha, const float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    y[idx] = alpha * x[idx] + beta;
}
```

1.  编写`iterative_saxpy_kernel()`函数，代码如下：

```cpp
__global__ void
iterative_saxpy_kernel(float *y, const float* x, 
                       const float alpha, const float beta, 
                       int n_loop)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < n_loop; i++)
        y[idx] = alpha * x[idx] + beta;
}

```

1.  编写`recursive_saxpy_kernel()`函数，代码如下：

```cpp
__global__ void
recursive_saxpy_kernel(float *y, const float* x, 
                       const float alpha, const float beta, 
                       int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (depth == 0)
        return;
    else
        y[idx] = alpha * x[idx] + beta;

    if (idx == 0)
        vecAdd_kernel_C<<< gridDim.x, blockDim.x 
                           >>>(y, x, alpha, beta, depth - 1);
}
```

1.  编写启动这些 CUDA 内核函数的主机代码。首先，我们将对`simple_saxpy_kernel()`函数进行迭代调用：

```cpp
for (int i = 0; i < n_loop; i++) {
    simple_saxpy_kernel<<< dimGrid, dimBlock >>>(
                           d_y, d_x, alpha, beta);
}
```

其次，我们将调用`iterative_saxpy_kernel()`内核函数，该函数内部有一个迭代循环：

```cpp
iterative_saxpy_kernel<<< dimGrid, dimBlock >>>(
                          d_y, d_x, alpha, beta, n_loop);
```

最后，我们将调用`recursive_saxpy_kernel()`内核函数，该函数以递归方式调用自身：

```cpp
recursive_saxpy_kernel<<< dimGrid, dimBlock >>>(
                          d_y, d_x, alpha, beta, n_loop);
```

循环次数小于或等于 24，因为最大递归深度为 24。除了简单的循环操作外，您不必在主机上放置循环操作，因为它已在内核代码中定义。

1.  使用以下命令编译代码：

```cpp
$ nvcc -run -m64 -gencode arch=compute_70,code=sm_70 -I/usr/local/cuda/samples/common/inc -rdc=true -o cuda_kernel ./cuda_kernel.cu
```

您必须使用 GPU 的计算能力版本号来选择`gencode`选项。

1.  测试编译后的应用程序。这个结果是使用 Tesla P40 测量的，因为 CUDA 9.x 不支持 Volta GPU 的 CUDA 动态并行性（CDP）配置文件：

```cpp
Elapsed Time...
simple loop: 0.094 ms
inner loop : 0.012 ms
recursion : 0.730 ms
```

# 三种执行的比较

从结果中，我们可以确认内部循环是迭代操作中最快的方法。以下截图显示了这个示例应用程序的分析结果：

![](img/0040c495-f138-4acc-bf8a-c2e2b02ebbda.png)

迭代内核调用显示了每个内核调用的内核启动开销。GPU 需要从设备内存中获取所有所需的数据，并需要调度 GPU 资源等。另一方面，内部循环内核显示了一个打包操作，因为所有所需的资源都是预先定位的，不需要重新调度其执行。由于我们之前讨论的动态并行性限制，递归内核操作显示了最长的执行时间。

一般来说，建议使用开销最小的方法。然而，很难说哪种内核调用设计优于其他，因为算法和问题比我们在这里涵盖的要多。例如，CDP 用于增强某些情况下的并行性，比如用于 GPU 树和搜索。

# 总结

在本章中，我们涵盖了几种内核执行机制。我们讨论了 CUDA 流是什么，以及如何使用它们同时执行多个内核函数。通过利用主机和 GPU 之间的异步操作，我们学到可以通过数据传输和内核执行来隐藏内核执行时间。此外，我们可以使用回调函数使 CUDA 流调用主机函数。我们可以创建一个有优先级的流，并确认其有优先级的执行。为了测量内核函数的确切执行时间，我们使用了 CUDA 事件，并且我们也学到 CUDA 事件可以用于与主机同步。在最后一节中，我们还讨论了每种内核执行方法的性能。

我们还涵盖了其他内核操作模型：动态并行性和网格级协作组。动态并行性使得内核函数内部可以进行内核调用，因此我们可以使用递归操作。网格级协作组实现了多功能的网格级同步，我们讨论了这个特性在特定领域的用途：图搜索、遗传算法和粒子模拟。

然后，我们扩展了我们对主机的覆盖范围。CUDA 内核可以从多个线程或多个进程中调用。为了执行多个线程，我们使用了带有 CUDA 的 OpenMP，并讨论了它的用处。我们使用 MPI 来模拟多进程操作，并且可以看到 MPS 如何提高整体应用程序性能。

正如我们在本章中看到的，选择正确的内核执行模型是一个重要的话题，线程编程也是如此。这可以优化应用程序的执行时间。现在，我们将扩展我们的讨论到多 GPU 编程来解决大问题。
