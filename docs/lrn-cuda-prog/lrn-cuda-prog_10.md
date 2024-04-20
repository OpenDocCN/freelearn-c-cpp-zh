# 第十章：使用 CUDA 加速深度学习

深度学习是一种可以根据人工神经网络解释数据的机器学习方法。具体来说，我们提供机器可以理解的数据，并构建学习数据表示的神经网络模型。我们可以使用这种技术构建识别语音、从图像中分类对象、理解文本、翻译语言、转换数据域等模型。基本的神经网络包括全连接层（FCL）、卷积神经网络（CNN）和循环神经网络（RNN）。这些架构在数据分类、区域理解和顺序关系方面显示出强大的准确性。

深度学习需要大量计算，以便广泛应用。然而，通过使用 GPU 计算能力，我们可以显著减少训练时间，从而解决了这个问题。这是因为神经网络的基本架构是基于矩阵运算的，而 GPU 是一个针对此进行了优化的硬件平台。具体来说，深度学习的创新是通过 NVIDIA CUDA 加速来解决的，因为深度学习中的许多算法可以加速。

在本章中，我们将简要回顾神经网络操作，并讨论如何在 GPU 上加速这些操作。作为实践，我们将使用 cuDNN 和 cuBLAS CUDA 库实现一个卷积网络。cuDNN 库是 NVIDIA 的 CUDA 库，专门优化了深度学习操作。我们将在三个部分中介绍其实现。我们还将介绍 GPU 如何优化所需的操作。然后，我们将通过比较**长短期记忆**（LSTM）网络的性能来介绍使用 cuDNN 库的有效性。然后，我们将介绍使用**NVIDIA 工具扩展**（NVTX）进行深度学习的性能分析。这可以测量 GPU 上的网络操作，以便我们可以分析时间线上的操作并了解其性能。

在本章中，我们将涵盖以下主题：

+   使用 CUBLAS 加速全连接层

+   使用 cuDNN 的逐元素层

+   cuDNN/CUDA 中的 Softmax 和损失函数

+   使用 cuDNN 的卷积神经网络

+   使用 CUDA 的循环神经网络

+   深度学习框架的性能分析

# 技术要求

本章需要安装 cuDNN 库和 CUDA 工具包。我们还需要 CUDA 启用的 GPU。本章将介绍深度学习的基础知识和性能，因此不需要新的 GPU 功能。换句话说，如果您已经涵盖了前几章的大部分内容，您将拥有一个适当的 GPU 来使用。

要安装 cuDNN 库，您需要从[`developer.nvidia.com/cudnn`](https://developer.nvidia.com/cudnn)下载软件包。您需要登录 NVIDIA 开发者网站才能访问下载页面。如果您还没有帐户，您需要注册一个 NVIDIA 开发者帐户。确保 cuDNN 与您安装的 CUDA 版本编译一致。

# 使用 cuBLAS 加速全连接层

全连接层是深度学习的基本架构。让我们回顾一下它的操作，并看看 CUDA 如何加速神经网络的前向和反向传播过程。然后，我们将把它们应用到 GPU 上。

# 神经网络操作

神经网络的基本操作是在输入数据和参数之间执行点操作。我们称之为感知。在深度学习中，神经网络以分层方式连接多个感知。我们称这些为前馈神经网络。以下图表显示了一个感知和基本神经网络：

![](img/451a22fa-7568-4602-a522-e2dd3826e53e.png)

感知器的基本操作是使用输入数据和适当的权重创建点积。然后，它使用激活函数进行非线性操作，例如 sigmoid 或**整流线性单元**（**ReLU**）。在前馈神经网络中，操作只是一个仿射变换，然后是激活函数的应用。一个向量将被馈送到神经网络作为输入，并与两层中每个节点之间的权重参数相乘。

为了训练神经网络，我们进行前向传播、损失计算和梯度反向传播，然后使用更新参数。让我们简要介绍一下它们。然后，我们将使用 cuBLAS 和其他 CUDA 操作来匹配每个步骤。

前向操作可以用以下方程表示：

![](img/3bbf27ea-c1c4-4091-bfca-a7ef3640f55d.png)

这里，![](img/f58e3de0-2d4a-43ab-888b-3ffe2c39d537.png) 是给定输入向量的预测结果，![](img/3b5c5481-dfe4-49e6-9bfd-16777e251f7f.png) 是权重参数矩阵，![](img/bf47e85d-b819-4362-ba4b-2c84785e42ce.png) 是激活函数。正如我们所看到的，全连接层中的基本操作是矩阵运算。因此，我们需要对输入和激活函数实现矩阵乘法运算。因为我们进行分类任务，所以我们使用 softmax 函数来规范化输出，并在下一层获得概率分布结果。

为了获得真实值之间的损失，我们对标签应用 one-hot 编码，并通过从每个元素获得熵来获得交叉熵损失，如下所示：

![](img/3ff26108-c02a-4604-9eaf-e1a5d04df24f.png)

我们可以通过每个交叉熵损失的总和来获得总损失值。然后，我们可以从前述方程中获得梯度。这看起来像一个复杂的操作，但可以简化如下：

![](img/fd0cf659-df61-472c-8fae-26facd67cc53.png)

现在，我们将梯度传播到前一层，这被称为反向传播。在这个任务中，我们使用链式法则来获得每个权重和偏差参数的梯度。然后，我们可以更新权重参数集和偏差。例如，我们可以通过以下方程获得权重和偏差的梯度：

![](img/78c891c9-a052-46b4-ba22-11229094a2dc.png)

我们可以通过以下方程获得梯度传播到前一层：

![](img/29f600b2-c4fd-4055-8b00-063de51c91a9.png)

这里，![](img/d6c48113-6009-476e-b53a-61484c12014e.png) 是激活函数的梯度。因此，我们需要从第二层获得![](img/9daa0c33-638e-4a0f-a64c-f6ad7d3b4b60.png) 用于第一层。然后，可以通过以下方程获得第一层的权重和偏差的梯度：

![](img/10ef2652-3473-4250-ab5a-08b258b45757.png)

现在，我们可以根据梯度下降规则更新权重和偏差，如下所示：

![](img/c829793e-26f4-4c44-8a50-d03e1c87a45b.png), ![](img/db73f6e5-a09f-4f6b-b850-9f78b7719aaa.png)

这里，![](img/1da8983e-1bb9-4ea6-9d8f-39426d0661f3.png) 是迭代步骤。

激活函数![](img/2eb7280e-6e41-45df-889f-049a2ed74da8.png)的梯度可能不同，其类型也可能不同。这个激活层的实现将在下一节中介绍。激活函数的导数可以用以下方程表示：

![](img/3d8453c6-78df-4735-85c0-6b13d98729a1.png), ![](img/da9de664-20a0-4af2-9b7a-95841c4fd8e6.png)

因此，神经网络操作是一组线性代数操作，并且可以使用 cuBLAS 库进行覆盖。实现的代码可以在`01_ann`中找到。我们将在*实现全连接层*、*实现层操作*和*实现 softmax 层*部分介绍这些实现细节。

# 神经网络层的设计

在编写代码之前，让我们来看看如何将操作打包成一个层配置：

1.  首先，我们执行前向操作。

1.  然后，我们执行反向操作。

1.  然后我们从梯度中得到一个权重更新。

1.  最后，输出层将获得损失。

这样，层可以配置如下：

![](img/b6bf914f-00bd-414e-b88f-bc36d05bb945.png)

它具有标准化的输入和输出，以及两种类型的输入，取决于工作流程。左侧数据路径将被命名为输入，而右侧将被命名为输出。数据分为两个阶段（前向和后向）。我们将使用 blob 来管理参数和输入/输出数据。blob 是跨层处理的数据的包装器，并帮助管理内存空间。我们将使用这种设计来简化网络的配置。每个层都将有每个 blob 的描述符和前向/后向处理操作。

现在，让我们创建一个层类，它将是所有层的基类。以下代码显示了`class`公共函数的堆叠。而且，你可以在`01_ann/src/ directory`的`layer.h`和`layer.cu`中找到它的实现。这不仅有前向和后向操作，还有权重更新控制和损失计算：

```cpp
class Layer
{
public:
    Layer();
    ~Layer();

    std::string get_name() { return name_; }

    virtual Blob<float> *forward(Blob<float> *input) = 0;
    virtual Blob<float> *backward(Blob<float> *grad_input) = 0;

    virtual float get_loss(Blob<float> *target);
    virtual int   get_accuracy(Blob<float> *target);

    void set_cuda_context(CudaContext *context) { cuda_ = context; }

    /* weights update control */
    void freeze() { freeze_ = true; }
    void unfreeze() { freeze_ = false;}
    void set_load_pretrain() { load_pretrain_ = true; }
    void set_gradient_stop() { gradient_stop_ = true; }
```

为了支持这些操作，层类维护了几个 cuDNN 描述符、blob 指针和权重更新控制器。当我们涵盖网络实现时，详细的实现将会被涵盖：

```cpp
protected:
    std::string name_;

    // Tensor descriptor for the input/output tensor
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    // filter and bias descriptor for weights and biases
    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t bias_desc_;

    // output memory
    Blob<float> *input_ = nullptr;       /* x */
    Blob<float> *output_ = nullptr;      /* y */
    Blob<float> *grad_input_ = nullptr;  /* dx */
    Blob<float> *grad_output_ = nullptr; /* dy */

    // master weights & bias
    bool freeze_ = false;               /* control parameter updates */
    Blob<float> *weights_ = nullptr;      /* w */
    Blob<float> *biases_  = nullptr;      /* b */
    Blob<float> *grad_weights_ = nullptr; /* dw */
    Blob<float> *grad_biases_  = nullptr; /* db */

    int batch_size_ = 0; // mini-batch size

    // cuda handle container
    CudaContext *cuda_ = nullptr;

    // initialize weights along with the input size
    void init_weight_bias(unsigned int seed = 0);
    void update_weights_biases(float learning_rate);

    // pretrain parameters
    bool load_pretrain_ = false;
    int load_parameter();
    int save_parameter();

    // gradient stop tagging
    bool gradient_stop_ = false;

    friend class Network;
}
```

这个层类将在其他部分的深度学习网络实现中使用。因此，它具有用于 cuDNN 操作的`cudnnTensorDescriptor_t`变量，以及`get_loss()`和`get_accuracy()`函数。

# 张量和参数容器

在我们的实现中，我们将使用一个名为`Blob`的数据容器。它的名称是从 Caffe 借来的。这使我们能够存储张量或网络参数以及其维度大小信息和内存点。我们将使用这个来连接每一层。这有助于每一层根据输入张量的大小信息初始化其权重。此外，每一层都可以根据`Blob`的信息验证其结果。

这个 blob 将需要神经网络中的维度大小信息，如下一行代码所示。然后，它的构造函数将根据大小信息创建一个主机端缓冲区：

```cpp
Blob<T>(int n, int c, int h, int w)
```

`Blob`还可以处理主机和设备上的内存，并帮助我们访问这些内存。`Blob`具有以下内存访问辅助函数：

```cpp
// get specified memory pointer
ftype *ptr() { return h_ptr_; }

// get cuda memory
ftype *cuda() 
{ 
    if (d_ptr_ == nullptr) 
        cudaMalloc((void**)&d_ptr_, sizeof(ftype) * len());
    return d_ptr_;
}

// transfer data between memory
ftype *to(DeviceType target) { 
    ftype *ptr = nullptr;
    if (target == host)
    {
        cudaMemcpy(h_ptr_, cuda(), sizeof(ftype) * len(), 
                   cudaMemcpyDeviceToHost);
        ptr = h_ptr_;
    }
    else // DeviceType::cuda
    {
        cudaMemcpy(cuda(), h_ptr_, sizeof(ftype) * len(), 
                   cudaMemcpyHostToDevice);
        ptr = d_ptr_;
    }
    return ptr;
}
```

正如我们之前讨论的，`Blob`可以存储张量，我们还需要提供张量形状信息，作为 cuDNN API 所需的描述符。因此，`Blob`可以使用以下代码创建和设置张量描述符：

```cpp
/* Tensor Control */
bool is_tensor_ = false;
cudnnTensorDescriptor_t tensor_desc_;
cudnnTensorDescriptor_t tensor()
{
    if (is_tensor_)
        return tensor_desc_;

    cudnnCreateTensorDescriptor(&tensor_desc_);
    cudnnSetTensor4dDescriptor(tensor_desc_, 
                                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                n_, c_, h_, w_);
    is_tensor_ = true;
    return tensor_desc_;
}
```

现在，让我们使用`Blob`来实现一个全连接层。

# 实现一个全连接层

在这一部分，我们将使用 cuBLAS 编写一个全连接网络。对于这个层，我们将创建一个从`Layer`类派生出来的`Dense`类。类构造函数将接收默认的层配置信息，如下所示：

```cpp
Dense::Dense(std::string name, int output_size)
{
    name_ = name;
    output_size_ = output_size;
}
```

但这还不足以配置整个层。缺失的信息将从输入中提供，因为输入大小将由前一层确定。现在，让我们来看看前向传播。

# 实现前向传播

在前向传播中，我们可以将前向过程分为两个步骤，如下所示：

![](img/95d7cd68-8fed-4f8a-bd7c-0f1d9e9a2afb.png)

由于权重大小不必受批量大小的影响，我们只考虑输入权重和输出权重的数量。另一方面，数据馈送 blob，如输入和输出，受批量大小的影响。因此，我们的 GEMM 操作与过滤器和输入数据可以设计如下：

![](img/7a96d4c2-0e5a-44ce-8251-e09d28e5e49a.png)

隐藏的输出将与偏置值相加。输入数据不仅限于数据加载器中的数据。当我们堆叠层时，上一层的输出将成为当前层的输入数据。前向操作可以实现如下：

```cpp
Blob<float> *Dense::forward(Blob<float> *input) {
  .. { blob initialization } ..

  // output = weights^T * input (without biases)
  cublasSgemm(cuda_->cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N, output_size_, 
        batch_size_, input_size_,
        &cuda_->one, weights_->cuda(), input_size_,
        input_->cuda(), input_size_,
        &cuda_->zero, output_->cuda(), output_size_);

  // output += biases * one_vec^T
  cublasSgemm(cuda_->cublas(), 
        CUBLAS_OP_N, CUBLAS_OP_N, output_size_, batch_size_, 1,
        &cuda_->one, biases_->cuda(), output_size_, one_vec, 1, 
        &cuda_->one, output_->cuda(), output_size_);
  return output_;
}
```

在第一次迭代中，每个层都需要初始化其权重和偏置。例如，这个`Dense`层可以初始化其权重、偏置和输出张量元素。我们可以将这个初始化任务分为两个阶段。第一个是权重和偏置，如下所示：

```cpp
// initialize weights and biases
if (weights_ == nullptr)
{
    // setup parameter size information
    input_size_ = input->c() * input->h() * input->w();

    // initialize weight, bias, and output
    weights_ = new Blob<float>(1, 1, input_size_, output_size_);
    biases_ = new Blob<float>(1, 1, output_size_);
}
```

接下来的阶段是关于更新输入信息和初始化输出 blob。当它是新的或需要重新配置时，我们需要做以下工作。在这个任务中，我们还需要创建一个填满我们批量大小的向量。这将用于偏置的添加：

```cpp
// initilaize input and output
if (input_ == nullptr || batch_size_ != input->n())
{
  input_ = input;
  batch_size_ = input->n();

  if (output_ == nullptr)
    output_ = new Blob<float>(batch_size_, output_size_);
  else
    output_->reset(batch_size_, output_size_);

  output_->tensor();

  if (d_one_vec != nullptr)
    cudaFree(d_one_vec);
  checkCudaErrors(cudaMalloc((void**)&d_one_vec, sizeof(float) * batch_size_));
  init_one_vec<<< (batch_size_+BLOCK_DIM_1D-1)/BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec, batch_size_);

  if (!freeze_)
    init_weight_bias();
}
```

这个初始化任务不仅触发了第一次迭代，还触发了批量大小的变化。在训练阶段不需要检查批量大小，但在测试阶段会很有用。这是因为训练和推断阶段的批量大小是不同的。在这种情况下，我们需要根据新的批量大小创建一个输出 blob。输出张量的大小是由通道大小确定的。以下代码创建了一个大小为（`batch_size_`，`output_size_`，`1`，`1`）的 blob：

```cpp
output_ = new Blob<float>(batch_size_, output_size_);
```

这将创建扁平化张量。然后，我们将馈送这些张量，这要求它们在通道中对齐。这种对齐在 softmax 层中是特别需要的。我们将在 softmax 层的实现中进行讨论。

在这个阶段的另一个重要任务是初始化权重和偏置。在我们的实现中，我们将使用 ReLU 作为激活函数。我们将使用正常的初始化器（[`arxiv.org/abs/1502.01852`](https://arxiv.org/abs/1502.01852)）技术使网络可训练。根据前述论文的指导，所需的权重值可以用以下方程生成：

![](img/85eca19f-7734-471a-b5a7-8191808d66f5.png)

![](img/0ed79072-61db-46e8-a6bc-66049715f0ff.png)是来自上一层的输入数量。因此，我们可以在更新输入张量信息后初始化参数。此外，偏置值将被初始化为`0`。以下代码显示了这一实现：

```cpp
void Layer::init_weight_bias(unsigned int seed)
{
    // Create random network
    std::random_device rd;
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>
                                        (seed));

    // He normal distribution
    float range = sqrt(6.f / input_->size());
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
        weights_->ptr()[i] = static_cast<float>(dis(gen));
    for (int i = 0; i < biases_->len(); i++)
        biases_->ptr()[i] = 0.f;

    // copy initialized value to the device
    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);
}
```

现在，让我们来讨论反向传播。

# 实现反向传播

正如我们之前讨论的，来自下一层的梯度被传播到这一层。基于传播的梯度，我们需要获得权重、偏置和数据（输入梯度）的三个梯度。我们需要创建可以存储它们的 blob。它们的大小不取决于批量大小，所以我们只需要确保创建它们。以下代码显示了我们如何为此目的创建 blob：

```cpp
if (grad_weights_ == nullptr) {
  grad_output_ = grad_output;
  grad_weights_ = new Blob<float>(weights_->shape());
  grad_biases_ = new Blob<float>(biases_->shape());
  grad_input_ = new Blob<float>(input_->shape());
}
```

在上述代码中，`grad_output_`表示从下一层传播的输出数据的梯度，`grad_input_`表示将传播到上一层的输入数据的梯度。因此，我们不需要创建`grad_output_` blob。如果您觉得这些命名约定令人困惑，也许更容易理解`grad_input_`为![](img/84d797c7-3cb7-40e4-8b12-bf9e5be2921e.png)，`grad_input_`为![](img/5dbe0798-7e12-4625-adcb-b2a5044495d2.png)。

以下代码显示了我们如何实现这一点：

```cpp
Blob<float> *Dense::backward(Blob<float> *grad_output) {
  .. { blob initialization } ..

  // db = (dy) * one_vec
  cublasSgemv(cuda_->cublas(),
    CUBLAS_OP_N,
    output_size_, batch_size_,
    &cuda_->one,
    grad_output_->cuda(), output_size_,
    one_vec, 1,
    &cuda_->zero,
    grad_biases_->cuda(), 1); 

  // dw = x * (dy)^T
  cublasSgemm(cuda_->cublas(),
    CUBLAS_OP_N, CUBLAS_OP_T,
    input_size_, output_size_, batch_size_,
    &cuda_->one,
    input_->cuda(), input_size_,
    grad_output_->cuda(), output_size_,
    &cuda_->zero,
    grad_weights_->cuda(), input_size_);

  // dx = W * dy
  if (!gradients_stop_)
    cublasSgemm(cuda_->cublas(),
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size_, batch_size_, output_size_,
      &cuda_->one,
      weights_->cuda(), input_size_,
      grad_output_->cuda(), output_size_,
      &cuda_->zero, 
      grad_input_->cuda(), input_size_);

  return grad_input_;
}
```

如果这一层是模型中的第一层，我们也可以跳过计算输入数据的梯度，因为我们不需要对其进行任何操作。

当我们想要更新权重时，将会更新权重和偏置值。在本节中，我们将使用**随机梯度下降**（**SGD**）来实现这一点。这个操作也可以在其他层中使用。在这里，我们将把这个函数放在`Layer`类中。权重更新也可以使用`cublas`函数来完成，如下所示：

```cpp
void Layer::update_weights_biases(float learning_rate)
{
  float eps = -1.f * learning_rate;
  if (weights_ != nullptr && grad_weights_ != nullptr) {
    // w = w + eps * dw
    cublasSaxpy(cuda_->cublas(),
      weights_->len(),
      &eps,
      grad_weights_->cuda(), 1,
      weights_->cuda(), 1);
  }

  if (biases_ != nullptr && grad_biases_ != nullptr)
  {
    // b = b + eps * db
    cublasSaxpy(cuda_->cublas(),
      biases_->b(),
      &eps,
      grad_biases_->cuda(), 1,
      biases_->cuda(), 1);
  }
}
```

正如你所看到的，我们可以使用学习率更新权重和偏差。当然，你也可以改变`eps`操作以应用其他优化算法。

# 层终止

在 C/C++编程中，程序员应该覆盖如何在终止类实例时返回所使用的资源。根据我们的设计，如果层具有权重参数并且可以从梯度中更新它们，该层最多会创建六个 blob。以下代码显示了终止 blob 的层终止代码，这些 blob 是在内部创建的：

```cpp
Layer::~Layer()
{
  if (output_ != nullptr) delete output_;
  if (grad_input_ != nullptr) delete grad_input_;

  if (weights_ != nullptr) delete weights_;
  if (biases_ != nullptr) delete biases_;
  if (grad_weights_ != nullptr) delete grad_weights_;
  if (grad_biases_ != nullptr) delete grad_biases_;
}
```

输入 blob 或张量描述符将由其他层或 blob 终止处理。层类是其他层的基类。因此，我们可以专注于终止自定义创建的资源，因为当我们终止任何派生层时，这个终止代码将一起被调用。

尽管我们已经设计了网络和层，但我们还应该开发一些额外的层来完成网络。例如，我们没有实现激活、softmax 和损失计算层。我们将在接下来的部分中介绍这些层。

# 使用 cuDNN 的激活层

神经网络层中有许多逐元素操作。激活函数是这些操作之一。cuDNN 库提供了六种激活函数：sigmoid、ReLU、tanh、clipped ReLU、ELU 和 identity。在 cuDNN 库中，`cudnnActivationForward()`执行前向操作，`cudnnActivationBackward()`执行后向操作。

让我们看一下`cuddnnActivationForward()`函数的接口，如下所示：

```cpp
cudnnStatus_t cudnnActivationForward( cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, 
    const void *x, const void *beta,  
    const cudnnTensorDescriptor_t yDesc, void *y)
```

使用`cudnnActivationDescriptor_t`，我们可以确定激活函数的类型。Alpha 和 beta 是标量值，用于确定要添加的输入速率。`xDesc`和`yDesc`保存张量的形状信息。它们可以使用`cudnnCreateTensorDescriptor()`创建。

当你看`cudnnActivationBackward()`函数时，`dy`是来自下一层的梯度输入，`dx`是输出到上一层的梯度。在这种情况下，`y`变成了输入。这样，`dyDesc`提供了梯度输入形状信息，而`dxDesc`提供了梯度输出形状信息：

```cpp
cudnnStatus_t cudnnActivationBackward( cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc,  
    const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc,  const void *x,
    const void *beta,  const cudnnTensorDescriptor_t dxDesc, void *dx)
```

一般来说，我们可以期望层之间的张量形状不会改变。因此，我们可以对`x`和`dx`使用相同的张量描述符。这与使用`y`和`dy`是一样的。

现在，让我们使用 cuDNN API 实现启用 cuDNN 的激活函数。要使用 cuDNN API，我们需要提供一个张量描述符来指定输入和输出张量的维度给 cuDNN 函数。我们还需要指定激活操作。

# 层配置和初始化

虽然我们的示例实现没有使用层接口，但我们需要将我们的示例集成到层接口中。在我们的层设计中，激活层可以这样实现：

```cpp
class Activation: public Layer
{
public:
  Activation(std::string name, cudnnActivationMode_t mode, 
             float coef = 0.f);
  ~Activation();

  Blob<float> *forward(Blob<float> *input);
  Blob<float> *backward(Blob<float> *grad_input);

private:
  cudnnActivationDescriptor_t act_desc_;
  cudnnActivationMode_t mode_;
  float coef_;
};
```

在初始化步骤中，我们需要创建几个张量描述符和一个激活描述符。cuDNN 库要求开发人员提供与 API 对应的张量大小或任何其他操作句柄：

```cpp
Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef)
{
  name_ = name;
  mode_ = mode;
  coef_ = coef;

  cudnnCreateActivationDescriptor(&act_desc_);
  cudnnSetActivationDescriptor(act_desc_, mode, CUDNN_PROPAGATE_NAN, coef);
}
```

在 cuDNN 中，我们使用激活描述符来指定激活函数操作。我们使用`cudnnSetActivationDescriptor()`函数来实现这一点。然后，它可以确定`cudnnActivationForward/Backward()`函数的操作。我们将在下一节中介绍这一点。然而，在这之前，我们需要实现类析构函数，以便它销毁激活描述符，如下所示：

```cpp
cudnnDestroyActivationDescriptor(activation_desc);
```

现在，让我们介绍激活层的前向和后向操作。

# 实现层操作

这也被称为警告操作。这个层不需要我们处理权重和偏差，因此比密集层更容易实现。

# 实现前向传播

在第一次迭代中，我们需要初始化输入描述符、输出描述符和输出 blob。当批处理大小改变时，我们将更新输出 blob。然而，我们不需要初始化权重和偏差，因为它们没有。以下代码显示了它的实现：

```cpp
if (input_ == nullptr || batch_size_ != input->n())
{
  input_ = input;
  input_desc_ = input->tensor();
  batch_size_ = input->n();

  if (output_ == nullptr)
    output_ = new Blob<float>(input->shape());
  else
    output_->reset(input->shape());

  output_desc_ = output_->tensor();
}
```

初始化后，我们使用 cuDNN 中的`cudnnActivationForward()`函数进行激活过程，如下所示：

```cpp
cudnnActivationForward(cudnnHandle, act_desc_, 
    &one, input_desc_, d_input, &zero, output_desc_, d_output);
```

这个激活函数的操作是在我们初始化这个层时确定的，正如我们之前讨论的。

# 实现反向传播

下一步是实现反向传播。我们将重用我们已经拥有的输入/输出张量描述符。现在，我们必须初始化我们希望反向传播的梯度：

```cpp
if (grad_input_ != grad_output_)
{
  grad_output_ = grad_output;
  grad_input_ = new Blob<float>(input_->shape());
  grad_input_->reset(input_->shape()); 
}
```

初始化后，我们可以调用`cudnnActivationBackward()`函数，如下所示：

```cpp
cudnnActivationBackward(cudnnHandle, activation_desc, 
    &one, output_desc_, output_->cuda(), output_desc_, 
    d_grad_output, input_desc_, input_->cuda(),
    &zero, input_desc_, grad_input_->cuda());
```

请注意，我们重用了在前向传递中创建的输入张量描述符和输出张量描述符。我们之所以能够这样做，是因为激活操作不会改变张量的大小。我们可以通过在激活反向传播中使用 cuDNN API 来简化我们的实现。

`cudnnActivationBackward()`函数的输出是`d_grad_input`。正如我们在前一节中描述的，这个梯度将传递给下一层。

现在，我们将实现 softmax 层，并将我们的层实现集成为一个网络。然后，我们将讨论图像分类任务中全连接层的准确性。

# cuDNN/CUDA 中的 softmax 和损失函数

对于 MNIST 数据集分类，我们将使用 softmax 分类器。softmax 函数对输入进行归一化，并生成![](img/6a23c4d0-c4cb-4caa-b483-128315b59c21.png)概率的概率分布。softmax 操作可以表示如下：

![](img/f6b6d132-a44b-4e74-ade8-f2dd5d043ea5.png)

cuDNN 的 softmax 前向函数支持此操作，以及通道和所有实例。之前，我们将密集层的输出与通道对齐。因此，我们将沿着通道应用 softmax 操作。

为了确认我们的训练有效完成，我们需要计算损失函数。由于 softmax 损失函数用于获取跨![](img/0adcd213-509d-4129-8488-317b8bc434e8.png)概率的损失，所以 softmax 损失函数被称为交叉熵损失。损失函数如下：

![](img/28d7a607-4b3b-400a-9382-74a6107c6d57.png)

我们需要获得这个 softmax 损失的梯度以更新神经网络。幸运的是，softmax 损失的梯度在求导后很简单，如下所示：

![](img/07eee5af-eac3-444a-b542-9acdf05f231e.png)

对于前向操作，我们将使用 cuDNN 函数来获取 softmax 的输出。为了获得梯度，拥有自定义操作更直观和简单。

# 实现 softmax 层

现在，让我们看看如何使用 cuDNN 和 CUDA 代码来实现 softmax 层。

# 实现前向传播

我们可以使用 cuDNN 库中的`cudnnSoftmaxForward()`来获得 softmax 成本函数的输出：

```cpp
cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, 
      CUDNN_SOFTMAX_MODE_CHANNEL,
      &one,  input_desc,  d_input, &zero, output_desc, d_output);
```

在这种情况下使用的最重要的参数设置之一是`CUDNN_SOFTMAX_MODE_CHANNEL`。此选项使得在输入张量描述符信息后面进行通道级别的 softmax 操作。通过这样做，我们可以提供已经通过密集层的小批量输入按通道对齐的张量。

# 实现反向传播

softmax 层的反向传递与其他层的实现不同。这个操作将输入数据的标签作为输入，并获得适当的梯度。正如我们之前讨论的，softmax 损失的梯度可以使用以下方程获得：

![](img/4e9ecd44-3c23-4f2a-a26b-2e69c5e10894.png)

我们可以使用`cublasSaxpy()`来实现这个操作，如下所示：

```cpp
// set grad_input_ as predict
cudaMemcpyAsync(grad_input_->cuda(), output_->cuda(), 
                output_->buf_size(), cudaMemcpyDeviceToDevice));
// set grad_input_ = predict - target 
cublasSaxpy(cuda_->cublas(), target->len(), &cuda_->minus_one,
            target->cuda(), 1, grad_input_->cuda(), 1));
```

在前面的代码中，目标 blob 包含了 one-hot 编码的目标向量，因此将负目标向量添加到预测值中会产生适当的梯度。之后，我们需要在传播到前一层之前对批次梯度进行归一化，如下所示：

```cpp
int grad_output_size = target->n() * target->c() * target->h() * target->w();
float scale = 1.0f / static_cast<float>(target->n());
cublasSscal(cuda_->cublas(), grad_output_size, &scale, grad_input_->cuda(), 1);
```

由于这引入了加权和的均值，我们可以期望每个批次的梯度被归一化。

# 实现损失函数

计算 softmax 的损失值是可选的。这意味着它的值在训练和推断中不被考虑。然而，我们可以将其用作训练的指标。

如我们之前讨论的，softmax 损失函数应该实现以下方程：

![](img/9e2f16bd-b541-4709-9d49-02e21c9a5aed.png)

我们可以通过一个核函数从每个样本的输出中获得损失并累积它们，如下所示：

```cpp
__global__ void
softmax_loss_kernel(float *reduced_loss, float *predict, 
                    float *target, int size)
{
  int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ float s_data[];
  float loss = 0.f;

  // each thread calculate entropy for each data 
  // and accumulate to shared memory
  if (batch_idx > 0)
    return;

  for (int c = 0; c < num_outputs; c++)
    loss += target[batch_idx * num_outputs + c] * \
                logf(predict[batch_idx * num_outputs + c]);
                workspace[batch_idx] = -loss;

  // Then, we do reduction the result to calculate loss 
  // Using 1 thread block
  if (blockIdx.x > 0) return;

  // Cumulate workspace data
  s_data[threadIdx.x] = 0.f;
  for (int i = 0; i < batch_size; i += blockDim.x)
    s_data[threadIdx.x] += workspace[threadIdx.x + i];

  __syncthreads();

  // Reduction
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (threadIdx.x + stride < batch_size)
      s_data[threadIdx.x] += s_data[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    reduced_loss[blockIdx.x] = s_data[0];
}
```

这个操作使用并行归约，在第三章 *CUDA 线程编程*中介绍过，用于获取一个批次中的累积损失值。由于我们只会使用这个减少的损失值来确认训练，所以我们只会监视它的输出而不是取平均值。

现在，让我们将我们实现的所有层与 MNIST 数据集加载器集成在一起。

# MNIST 数据加载器

整个过程中一个重要的部分是为特定数据集创建一个数据加载器。在这个实验室中，我们将使用包含 60,000 个样本的 MNIST 数据集。在初始化时，我们告诉数据加载器它应该加载训练集还是测试集。之后，数据加载器将加载数据集中的一些魔术数字，以及所有样本和它们的标签。加载的数据将被存储在向量中，并使用相同的随机种子进行洗牌。由于数据加载器构建和洗牌样本向量，训练循环或测试循环可能会在每次迭代时获得随机化的输入数据。完整的实现代码可以在本书的 GitHub 存储库中的`src/mnist.cpp`文件中找到。

# 管理和创建模型

当我们有多个层时，我们需要一个可以管理这些层的对象，进行神经网络操作，即前向/后向传播和权重更新。在这个实验室中，我们将有一个层的数组，并迭代数组进行前向处理。例如，前向操作可以用以下代码执行：

```cpp
Blob<float> *Network::forward(Blob<float> *input) {
  output_ = input;
  for (auto layer : layers_)
    output_ = layer->forward(output_);

  return output_;
}
```

反向传播也可以通过以相反顺序迭代数组来完成：

```cpp
void Network::backward(Blob<float> *target) {
  Blob<float> *gradient = target;
  // back propagation.. update weights internally.....
  for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
    // getting back propagation status with gradient size
    gradient = (*layer)->backward(gradient);
  }
}
```

如您所见，我们在向量中管理层，并具有每个层的操作。将新层添加到网络中甚至更简单，如下面的代码所示：

```cpp
void Network::add_layer(Layer *layer) {
  layers_.push_back(layer);
}
```

通过使用`Network`类，我们可以使用各种模型管理函数，如参数更新，层注册，层初始化等。此外，我们可以构建一个像现代深度学习框架一样的神经网络。例如，我们可以创建一个模型如下：

```cpp
// step 1\. loading dataset
MNIST data_loader = MNIST("./dataset");
// create training dataset loader and shuffling the data
data_loader.train(batch_size, true);  

// step 2\. model initialization
Network model;
model.add_layer(new Dense("dense1", 500));  // 1st layer
model.add_layer(new Dense("dense2", 10));   // 2nd layer
model.cuda();     // set cuda context for each layer
```

我们还可以有以下训练循环：

```cpp
// get data sample's shared buffer
Blob<float> *train_data   = data_loader.get_data();   
// get target's shared buffer
Blob<float> *train_target = data_loader.get_target(); 
// load data and targets with the batch size
data_loader.get_batch();    
tp_count = 0;  step = 0;
while (step < num_steps)
{
  // transfer loaded data to the GPU
  train_data->to(cuda);
  train_target->to(cuda);

  model.forward(train_data);    // forward
  model.backward(train_target); // backward
  learning_rate *= 1.f / (1.f + lr_decay * step);
  model.update(learning_rate);  // update

  step = data_loader.next(true); // load next data

  ... monitoring logic ...
}
```

对于测试阶段，我们为测试数据集创建另一个数据集加载器，并只进行前向传播的迭代。以下代码显示了它的实现：

```cpp
test_data_loader.test(batch_size_test);                   // create test dataset loader
Blob<float> *test_data = test_data_loader.get_data();     // get sample data shared buffer
Blob<float> *test_target = test_data_loader.get_target(); // get target shared buffer
test_data_loader.get_batch();    // load samples and targets with the batch size
tp_count = 0; step = 0;
while (step < num_steps_test) {
  // transfer loaded data to the GPU
  test_data->to(cuda);
  test_target->to(cuda);

  model.forward(test_data);  // forward
  tp_count += model.get_accuracy(test_target);

  step = test_data_loader.next(); // load next data
}
float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;
```

在测试阶段，我们将在完成对测试数据集中所有样本的测试后获得准确率。现在，我们需要在测试循环之后获得准确率。

# 使用 MNIST 数据集进行网络训练

现在，让我们运行我们实现的代码并查看其结果。对于训练阶段，我们将迭代 2,400 步，批量大小为 256。MNIST 数据集在训练集中有 60,000 个样本。2,400 步意味着我们将进行大约 10 个 epochs 的迭代。样本代码可以用以下命令编译：

```cpp
$ nvcc -run -m64 -std=c++11 -I/usr/local/cuda/samples/common/inc -gencode arch=compute_70,code=sm_70 -lcublas -lcudnn -lnvToolsExt -o train ./train.cpp ./src/layer.cu ./src/loss.cu ./src/mnist.cpp ./src/network.cpp
```

以下截图显示了我们实现的训练和测试输出：

![](img/1308649b-62e6-4f67-a98f-876d835049b2.png)

在训练迭代中，网络从训练数据集中获得了 92%的准确率。然而，测试准确率只有 77%，这与训练结果相比是一个相对较低的分数。推断显示训练和推断之间的准确率差距很大可能有很多原因。一个可能的原因是全连接层没有考虑到前面截图中显示的区域信息。在深度学习中，我们使用卷积层来使网络学习空间信息。

现在，让我们使用 cuDNN 实现卷积层，将其添加到网络中，并比较模型的性能。

# 使用 cuDNN 的卷积神经网络

cuDNN 库为卷积操作提供了优化的性能。通过创建一个卷积层，我们将覆盖 API 的配置，用于前向和后向操作。

卷积网络层对输入数据进行卷积处理。当你想要构建一个了解区域信息的神经网络时，这种网络架构是很有用的。回想一下，在第七章中的卷积实现，*CUDA 中的并行编程模式*，它需要相当大的内存带宽，并需要进一步优化以获得最佳性能。然而，使用 cuDNN 库，我们也可以获得最佳性能，因为我们不必重新发明轮子。

卷积层的实现与全连接层的实现类似。然而，由于 cuDNN 库的存在，有两个不同之处：我们不必像以前那样完全实现那么多细节，我们需要为操作分配一个工作空间大小。对于每个卷积操作——前向、反向滤波器和反向输入——都需要额外的内存空间，取决于它们的算法。算法可以根据给定的输入/输出/滤波器张量维度而变化。详细的 API 调用将在稍后处理。

与其他层一样，它有三个工作阶段。对于推理阶段，我们将调用`cudnnConvolutionForward()`和`cudnnAddTensor()`。对于反向阶段，我们将调用`cudnnConvolutionBackwardData()`、`cudnnConvolutionBackwardFilter()`和`cudnnConvolutionBackwardBias()`。最后，对于更新阶段，我们可以重用全连接层的代码。该层的配置概述如下：

实现前向传播

在深度学习神经网络中，通常会与卷积网络一起使用池化层。池化层只是根据简单的规则选择输入数据进行输出。以下图示显示了最大池化的例子：

![](img/e821b3b9-c7ff-4bdc-817a-1b7c5fb97325.png)

使用 cuDNN 库，我们将实现这两个卷积操作。

# 卷积层

与全连接层类似，这个卷积层有权重和偏置参数。在全连接层中，我们使用了 cuBLAS，它不需要 cuDNN 相关的描述符。然而，我们将使用 cuDNN 卷积函数，因此需要使用滤波器描述符和卷积操作描述符。以下代码显示了在构建层时应该初始化的资源：

```cpp
Conv2D::Conv2D(std::string name,
        int out_channels, kernel_size, stride, padding, dilation):
        out_channels_(out_channels), kernel_size_(kernel_size),
        stride_(stride), padding_(padding), dilation_(dilation) {
  name_ = name;
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnSetConvolution2dDescriptor(conv_desc_,
    padding_, padding_, stride_, stride_, dilation_,dilation_,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
}
```

由于我们在模型构建时提供了卷积操作信息，我们可以指定卷积描述符。然而，滤波器的操作可以在推断时指定，因为我们可以在那时学习输入张量的大小。现在，让我们实现卷积层的前向传递。

# ![](img/dfbcecb1-0c13-4d9e-a0a3-ac60623461a2.png)

正如我们之前讨论的，我们可以用输入张量大小初始化卷积层。这个输入张量大小会影响输出张量的大小。以下代码显示了前向传递中的参数初始化步骤：

```cpp
// initialize weights and bias
if (weights_ == nullptr) {
  // initialize containers handles
  cudnnSetFilter4dDescriptor(filter_desc_, 
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    out_channels_, input->c(), kernel_size_, kernel_size_);

  weights_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
  biases_ = new Blob<float>(1, out_channels_); // bias size
  bias_desc_ = biases_->tensor();
}
```

然后，我们需要更新输入资源，初始化输出 blob，创建 cuDNN 工作空间，并初始化权重参数，如下所示：

```cpp
// initilaize input and output
if (input_ == nullptr || batch_size_ != input->n()) {
  // initialize input
  input_ = input;
  input_desc_ = input->tensor();
  batch_size_ = input->n();

  // getting output tensor size
  cudnnGetConvolution2dForwardOutputDim(
    conv_desc_, input_desc_, filter_desc_,
    &output_size_[0], &output_size_[1], 
    &output_size_[2], &output_size_[3]);

  // initialize output blob
  if (output_ == nullptr)
    output_ = new Blob<float>(output_size_);
  else
    output_->reset(output_size_);
  output_desc_ = output_->tensor();

  // initialize weights
  if (!freeze_)
    init_weight_bias();

  // initialize workspace for cudnn
  set_workspace();
}
```

为了获得输出张量大小，我们使用`cudnnGetConvolution2dForwardOutputDim()`函数。该函数根据输入张量大小、卷积操作和滤波器大小输出维度大小信息。然后，我们重用了在全连接层中使用的相同参数初始化代码。

要调用 cuDNN 的卷积 API，我们需要提供其工作算法和工作空间内存。我们这样做是因为 cuDNN 根据卷积大小选择最佳卷积算法，并且需要立即进行测量。确定算法后，cuDNN 可以确定工作空间大小。卷积层需要进行前向传播的卷积操作、输入数据的梯度和权重的梯度。我们需要分别处理每个算法，但我们可以分配一个工作空间，因为工作空间专门用于每个卷积操作。

因此，我们创建的工作空间需要具有每个卷积算法所需的最大大小。以下代码显示了我们如何使用它们并管理工作空间：

```cpp
Conv2d::set_workspace() {
  size_t temp_size = 0;

  // fwd
  cudnnGetConvolutionForwardAlgorithm(cuda_->cudnn(),
    input_desc_, filter_desc_, conv_desc_, output_desc_,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_fwd_algo_);
  cudnnGetConvolutionForwardWorkspaceSize(cuda_->cudnn(),
    input_desc_, filter_desc_, conv_desc_, output_desc_, 
    conv_fwd_algo_, &temp_size);
  workspace_size = std::max(workspace_size, temp_size);

  // bwd - data
  cudnnGetConvolutionBackwardDataAlgorithm(cuda_->cudnn(), 
    filter_desc_, output_desc_, conv_desc_, input_desc_, 
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, 
    &conv_bwd_data_algo_);
  cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->cudnn(),
    filter_desc_, output_desc_, conv_desc_, input_desc_, 
    conv_bwd_data_algo_, &temp_size);
  workspace_size = std::max(workspace_size, temp_size);

  // bwd - filter
  cudnnGetConvolutionBackwardFilterAlgorithm(cuda_->cudnn(),
    input_desc_, output_desc_, conv_desc_, filter_desc_,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, 
    &conv_bwd_filter_algo_);
  cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda_->cudnn(),
    input_desc_, output_desc_, conv_desc_, filter_desc_, 
    conv_bwd_filter_algo_, &temp_size);
  workspace_size = std::max(workspace_size, temp_size);

  if (workspace_size > 0) {
    if (d_workspace != nullptr)
      cudaFree(d_workspace);
    cudaMalloc((void**)&d_workspace, workspace_size);
  }
}
```

每个卷积算法都使用单独的类型进行指定，即`cudnnConvolutionFwdAlgo_t`、`cudnnConvolutionBwdDataAlgo_t`和`cudnnConvolutionBwdFilterAlgo_t`。我们可以通过将它们声明为类成员变量来使用它们，即`conv_fwd_algo_`、`conv_bwd_data_algo_`和`conv_bwd_filter_algo_`。

现在，在初始化后，我们编写前向处理代码。我们使用滤波器进行卷积并添加偏差。以下代码显示了 cuDNN 卷积前向实现：

```cpp
cudnnConvolutionForward(cuda_->cudnn(), &cuda_->one, input_desc_, input_->cuda(), \
    filter_desc_, weights_->cuda(), conv_desc_, conv_fwd_algo_, d_workspace, workspace_size, \
    &cuda_->zero, output_desc_, output_->cuda());
cudnnAddTensor(cuda_->cudnn(), &cuda_->one, bias_desc_, biases_->cuda(), \
    &cuda_->one, output_desc_, output_->cuda());
```

卷积的结果将使用输出 blob 传递到下一层。

# 实现反向传播

在反向传播中，我们应该计算偏差的梯度、权重的梯度和输入数据的梯度。为此，我们需要在第一次迭代中创建 blob 以便我们可以存储它们。它们的大小不取决于批处理大小，所以我们只需要确保它们被创建。初始化步骤可以实现如下：

```cpp
// initialize grad_output back-propagation space
if (grad_weights_ == nullptr) {
  grad_output_  = grad_output;
  grad_weights_ = new Blob<float>(weights_->shape());
  grad_biases_  = new Blob<float>(1, biases_->c());
  grad_input_   = new Blob<float>(input_->shape());
}
```

然后，我们调用 cuDNN 反向卷积 API，如下所示：

```cpp
Blob<float> *Conv2D::backward(Blob<float> *grad_output) {
  ... { initialization step } ...

  // gradients of biases
  cudnnConvolutionBackwardBias(cuda_->cudnn(),
    &cuda_->one, 
    output_desc_, grad_output->cuda(),
    &cuda_->zero, 
    bias_desc_, grad_biases_->cuda());

  // gradients of weights 
  cudnnConvolutionBackwardFilter(cuda_->cudnn(),
    &cuda_->one, 
    input_desc_, input_->cuda(), 
    output_desc_, grad_output_->cuda(),
    conv_desc_, conv_bwd_filter_algo_, d_workspace, workspace_size,
    &cuda_->zero, 
    filter_desc_, grad_weights_->cuda());

  // gradients of input data
  if (!gradient_stop_)
    cudnnConvolutionBackwardData(cuda_->cudnn(),
      &cuda_->one, 
      filter_desc_, weights_->cuda(), 
      output_desc_, grad_output->cuda(), 
      conv_desc_, conv_bwd_data_algo_, d_workspace, workspace_size,
      &cuda_->zero, 
      input_desc_, grad_input_->cuda());
```

然后，我们将输入数据的梯度传递给前一层以传播梯度。在更新步骤中，我们将使用基类的梯度更新代码来更新权重和偏差的梯度。在全连接层中实现反向传播时，我们已经涵盖了这一点。如果这是第一层，则我们也可以跳过计算输入数据的梯度。

# 使用 cuDNN 的池化层

池化层有两个特点。首先，它的输出大小与卷积层不同，cuDNN 为此提供了相应的 API。其次，它没有任何内部权重。

为了指定池化操作，我们可以使用 cuDNN 的`cudnnPoolingDescriptor_t`函数，并在类构造函数中创建和指定 cuDNN 的池化描述符，如下所示：

```cpp
cudnnCreatePoolingDescriptor(&pool_desc_);
cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
  kernel_size_, kernel_size_, padding_, padding_, stride_, stride_);
```

现在，让我们实现池化层的前向和反向操作。

# 实现前向传播

池化层有助于减小张量的大小。因此，我们需要计算输出大小。我们可以使用`cudnnGetPooling2dForwardOutputDim()`函数来计算大小，就像我们在卷积层实现中所做的那样。此外，张量大小取决于批处理大小。这意味着如果批处理大小发生变化，我们需要更新张量大小。以下代码显示了我们如何初始化输入和输出 blob：

```cpp
if (input_ == nullptr || batch_size_ != input->n()) {
  input_ = input;

  // resource initialize
  input_desc_ = input_->tensor();
  batch_size_ = input->n();

  // setting output
  cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_, 
    &output_size_[0], &output_size_[1], &output_size_[2], 
    &output_size_[3]);
  if (output_ == nullptr)
    output_ = new Blob<float>(output_size_);
  else
    output_->reset(output_size_);

  output_desc_ = output_->tensor();
}
```

对于前向传播，我们调用`cudnnPoolingForward()`函数，如下所示：

```cpp
Blob<float> *Pooling::forward(Blob<float> *input) {
  ... { initialization step } ...

  cudnnPoolingForward(cudnnHandle, pool_desc_, &one, 
    input_desc_, input_->cuda(),
    &zero, output_desc_, output_->cuda());
}
```

# 实现反向传播

对于反向传播步骤，我们调用`cudnnPoolingBackward()`函数，如下所示：

```cpp
Blob<float> *Pooling::backward(Blob<float> *grad_output) {
  if (grad_input_ == nullptr)
    grad_input_ = new Blob<float>(input_->shape());

  cudnnPoolingBackward(cudnnHandle, pool_desc_,
    &one, output_desc_, output_->cuda(), 
    output_desc_, grad_output->cuda(), 
    input_desc_, input_->cuda(), 
    &zero, input_desc_, grad_input_->cuda());
}
```

池化层的张量形状的输入和梯度的输入是相同的，输出和梯度的输出的形状也是相同的。因此，我们可以分别重用输入和输出的张量描述符。

现在，让我们将这些集成到单个卷积层实现中。

# 网络配置

现在，我们将更新我们之前的网络 LeNet。网络代码可以编写如下：

```cpp
Network model;
model.add_layer(new Conv2D("conv1", 20, 5));
model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
model.add_layer(new Conv2D("conv2", 50, 5));
model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
model.add_layer(new Dense("dense1", 500));
model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
model.add_layer(new Dense("dense2", 10));
model.add_layer(new Softmax("softmax"));
model.cuda();
```

现在，我们可以开始训练和推断阶段，因为我们已经配置了我们的层，使它们彼此连接。让我们使用以下命令编译代码：

```cpp
$ nvcc -run -m64 -std=c++11 -I/usr/local/cuda/samples/common/inc -gencode arch=compute_70,code=sm_70 -lcublas -lcudnn -lnvToolsExt -o train ./train.cpp ./src/layer.cu ./src/loss.cu ./src/mnist.cpp ./src/network.cpp
```

然后，我们可以看到训练和测试结果如下：

![](img/7bddab01-eac8-4ff6-9222-861e7e99c72a.png)

正如您所看到的，该网络的训练准确度和推断准确度都比仅使用全连接网络时要高。我们还可以通过查看 NVIDIA 配置文件来确认其操作，如下所示：

![](img/60c3e92b-b87f-49e5-8a22-012f0da4be45.png)

# 混合精度操作

最新的 NVIDIA GPU 支持深度学习的混合精度操作。我们不会在本书中涵盖这一点，因为它超出了我们的范围。但是，如果您希望了解更多，可以访问 NVIDIA 提供的示例，位于`/usr/src/cudnn_samples_v7/conv_sample`。要访问此示例，您需要从 cuDNN 网页下载示例。此示例代码显示了如何使用 cuDNN 库进行混合精度操作。

为了使 cuDNN API 与张量核心一起工作，我们需要设置数学类型，如下所示：

```cpp
cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH);
```

然后，我们需要使用`cudnnSetTensorNdDescriptor()`初始化输入和输出张量的张量描述符。这为张量提供填充，以便我们获得优化的张量核心性能。

一个很好的基于 cuDNN 的实现是`cudnn-training`：[`github.com/tbennun/cudnn-training`](https://github.com/tbennun/cudnn-training)。它将 LeNet 实现为一系列 cuDNN 函数。您可以跟踪每一行，看看 CUDNN 函数是如何工作的。

如果您有兴趣使用 cuDNN 部署您的网络，请查看以下关于 GTC-CNN 推断与 cuDNN 的视频（[`developer.nvidia.com/gtc/2019/video/S9644/video`](https://developer.nvidia.com/gtc/2019/video/S9644/video)）。这个讲座介绍了使用 cuDNN 进行 CNN 推断的有用性能优化技巧。

在深度学习训练中使用半精度需要超过 FP16 操作的利用率。我们需要在 FP16 中计算张量，同时将权重保持在 FP32 中。此外，一些操作需要 FP32。我们称之为混合精度。cuDNN 库提供了一个名为 mnistCUDNN 的混合精度推断示例。该示例显示了输入和层数据类型的转换。如果您想了解更多关于深度学习和训练中混合精度操作的信息，请阅读以下文章：[`devblogs.nvidia.com/video-mixed-precision-techniques-tensor-cores-deep-learning/`](https://devblogs.nvidia.com/video-mixed-precision-techniques-tensor-cores-deep-learning/)。

现在，我们将从性能方面讨论深度学习中的其他 GPU 使用注意事项。

# 循环神经网络优化

RRN 允许您在深度学习中分析顺序数据。尽管该网络具有顺序依赖性，但仍有大量的优化空间。在本节中，我们将介绍其算法以及 cuDNN 如何提供优化性能。

有许多种类型的 RNN，但 cuDNN 只支持四种，即带有 ReLU 的 RNN，带有 tanh 的 RNN，LSTM 和 GRU。它们有两个输入：来自先前网络的隐藏参数和来自源的输入。根据它们的类型，它们有不同的操作。在本实验室中，我们将介绍 LSTM 操作。下图显示了 LSTM 的前向操作：

![](img/674b1b76-8aa5-44b7-a7cd-dada656f74b0.png)

从计算的角度来看，有八个矩阵-矩阵乘法和许多逐元素操作。根据这个估计，我们可以期望 LSTM 可能是内存受限的，因为每个操作都是内存受限的。另一方面，CUDNN 提供了`cudnnRNNForwardInference()`和`cudnnRNNFowardTraining()`RNN 函数。我们将通过测量这个函数的性能和模拟 LSTM 的性能来介绍使用这个函数的好处。为了做到这一点，我们将实现一个虚拟的 LSTM 层，并将其性能与 cuDNN LSTM 函数进行比较。

为了测试目的，我们将设置超参数如下：

```cpp
int mode = 2; // LSTM in CUDNN
int seq_length = 512;
int num_layers = 4;
int hidden_size = 512;
int input_size = hidden_size;
int batch_size = 32;
float dropout_rate = 0;
bool bidirectional = 0;
int persistent = 0;
```

序列长度或隐藏大小可能会有所不同，这取决于问题。在这个测试中，我们将使用`512`作为长度，在序列研究中经常使用。CUDNN API 需要更多的选项才能工作，比如 dropout 率、双向或单向以及持久 RNN。在本节中，我们只测试 vanilla LSTM。

# 使用 CUDNN LSTM 操作

让我们编写一些执行`cudnnRNNForwardTraining()`函数作为 LSTM 层的代码：

1.  我们需要初始化输入和输出内存空间。为了执行 cuDNN 的 RNN API，我们需要使用以下变量：

```cpp
// hx, cx, hy, cy, dhy, dcy, dhx, and dcs can be null.
void *x;            // input
void *hx = nullptr; // input of initial hidden state
void *cx = nullptr; // input of cell state (LSTM)

void *y;            // output
void *hy = nullptr; // output of final hidden state
void *cy = nullptr; // output of final cell state (LSTM)

void *dy;            // input of gradient 
void *dhy = nullptr; // input of final hidden state
void *dcy = nullptr; // input of final cell state (LSTM)

void *dx;            // output of gradient at the input of rnn
void *dhx = nullptr; // output of gradient at the initial hidden state
void *dcx = nullptr; // output of gradient at the initial cell state
```

这些变量是 LSTM 的输入和输出。为了提供输入和获取输出，我们需要分配适当的内存空间。根据 LSTM 的定义，我们需要考虑输入、输出和隐藏层的长度。这些大小可以确定如下：

```cpp
int input_length = seq_length * input_size * batch_size;
int output_length = seq_length * hidden_size * batch_size;
int hidden_length = hidden_size * batch_size * num_layers;
```

然后，我们可以为每个项目分配内存。

1.  现在，我们需要为 cuDNN RNN API 设置张量描述符。以下代码显示了我们应该设置的所需张量描述符：

```cpp
cudnnTensorDescriptor_t x_desc[seq_length], y_desc[seq_length], \
                        dx_desc[seq_length], dy_desc[seq_length];
cudnnTensorDescriptor_t hx_desc, cx_desc;
cudnnTensorDescriptor_t dhx_desc, dcx_desc;
cudnnTensorDescriptor_t hy_desc, cy_desc;
cudnnTensorDescriptor_t dhy_desc, dcy_desc;
```

对于输入和输出描述符，我们需要初始化每个元素，即批量大小和其输入大小。其他隐藏的张量描述符是用层数、批量大小和隐藏大小进行初始化的。本节不涵盖如何编写初始化代码。但是，如果您想了解更多信息，可以查看`10_deep_learning/03_rnn`文件中的代码。

1.  我们还需要为 RNN 操作提供一个工作空间，就像我们为卷积操作做的那样：

```cpp
void *workspace;
cudnnFilterDescriptor_t w_desc, dw_desc;
cudnnSetRNNDescriptor_v6(cudnnHandle, rnn_desc,
                         hidden_size, num_layers, dropout_desc, CUDNN_LINEAR_INPUT,
                         bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                         CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
size_t weight_size;
cudnnGetRNNParamsSize(cudnnHandle, rnn_desc, x_desc[0], &weight_size, CUDNN_DATA_FLOAT);
cudaMalloc((void**)&workspace, weight_size);
```

然后，我们可以根据工作空间的大小设置滤波器描述符，如下所示：

```cpp
dimW = {weight_size / sizeof(float), 1, 1}
cudnnCreateFilterDescriptor(&w_desc);
cudnnCreateFilterDescriptor(&dw_desc);
cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW);
cudnnSetFilterNdDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW);
cudnnRNNForwardTraining(cudnnHandle, rnn_desc, seq_length,
                x_desc, x, hx_desc, hx, cx_desc, cx,
                w_desc, w, 
                y_desc, y, hy_desc, hy, cy_desc, cy,
                workspace, workspace_size, reserved_space, 
                reserved_size);
```

我们可以使用`cudaEvnetRecoard()`和 flops 计算来衡量它们的性能。例如，前向操作可以配置为以下方程：

![](img/b385c8ec-d143-4f0e-9cc6-d31523159f38.png)

然后，我们将通过将批量大小从 32 增加到 256 来测试我们的实现，每次增加 32。适用的测试范围可能会有所不同，以及 GPU 的内存大小。

在本节中，我们实现了基于 LSTM 的模拟和`cudnnRNNForwardTraining()`调用。我们部分模拟的版本只有 GEMM 操作，这是最计算密集的操作。现在，让我们比较这些实现的性能。

# 实现虚拟 LSTM 操作

在我们的实现中，我们将专注于模拟 LSTM 的主要操作，而不是完全实现它。

让我们确定 LSTM 网络的超参数。一般来说，输入序列长度范围从 512 到 2,048。层数的数量是不确定的。但是，由于*tanh*操作，它不能太大。对于输入大小，我们将使用 512。通常情况下，批量大小在 RNN 使用方面在 32 到 256 之间。CUDNN 需要更多关于 dropout 率、双向或单向以及是否使用持久 RNN 的输入。我们现在不使用它们。我们的 LSTM 配置信息如下：

![](img/652a450a-efec-4f75-9b83-4048e9751bc0.png)

现在，我们将部分实现 LSTM 操作以测量计算强度。正如我们之前讨论的，LSTM 有两个矩阵-矩阵乘法需要计算。LSTM 操作将为输入序列的每个元素以及每个层计算。然后，操作可以配置如下：

```cpp
for (int layer = 0; layer < num_layers; layer++) {
  for (int linear_layer = 0; linear_layer < 4; linear_layer++) {
    for (int sequence = 0; sequence < seq_length; sequence++) {
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
      hidden_size, input_size, batch_size,
      &alpha, input_weight, input_size, x, input_size,
      &beta, h, hidden_size);
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
      hidden_size, hidden_size, batch_size,
      &alpha, recurrent_weight, hidden_size,
      h, hidden_size,
      &beta, y, hidden_size);
    }
  }
}
```

我们可以使用更多的逐元素操作，但这只是近似计算强度，所以我们暂时不考虑它们。

# 比较 CUDNN 和 SGEMM LSTM 的性能

让我们比较它们的性能以及不同的批处理大小，如下所示的代码实现在`main()`函数中：

```cpp
for (int step = 1; step <= 8; step++)
{
 batch_size = 32 * step;
 printf("Batch Size: %3d\n", batch_size);
 rnn_operation(seq_length, num_layers, hidden_size, input_size,   
   batch_size, dropout_rate, bidirectional, mode, persistent);
 cublas_operation(mode, 2ull, input_size, hidden_size, seq_length, batch_size, num_layers);
}
```

然后，我们可以使用以下命令编译和执行示例源代码：

```cpp
$ nvcc -run -m64 -std=c++11 -I/usr/local/cuda/samples/common/inc -gencode arch=compute_70,code=sm_70 -lcublas -lcudnn -lcurand -o rnn ./rnn.cpp
```

以下图表显示了来自 Tesla V100 卡的 cuBLAS 和 cuDNN 的性能：

![](img/cecc6413-86bd-45aa-ab52-03bb687493b1.png)

在上图中，两种实现在性能上有很大差异。cuDNN 的 LSTM 性能比使用 cuBLAS 模拟的 LSTM 要好得多。此外，LSTM 操作的性能遵循 Tesla V100 GPU 的屋顶线。另一方面，两个 SGEMM 操作并没有显示出这种性能，因为矩阵大小不够大以获得完整的性能。要从 Tesla V100 获得 10 TFlops，矩阵大小应与 1,024 的平方相似或更大。然而，正如我们所看到的，我们的矩阵大小大约是 512 的平方。

LSTM 优化在以下 NVIDIA 文章中有解释：[`devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5`](https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5)。它结合了矩阵-矩阵乘法，融合逐元素操作，多个流和多层并行化。

RNN 的优化版本之一是持久 RNN（[`svail.github.io/persistent_rnns`](https://svail.github.io/persistent_rnns)），由 Greg Diamos 介绍。尽管他的实现不包括 LSTM 和 GRU，但您可以了解 RNN 如何进行优化。

# 深度学习框架的性能分析

一般来说，我们使用 TensorFlow、PyTorch 和 MxNet 等深度学习框架开发和研究神经网络。由于这些框架，我们可以有效地开发复杂的模型。然而，当涉及性能工程时，由于性能分析工具的能力，理解框架下 GPU 操作是一个陡峭的学习曲线。例如，使用 Chrome 跟踪进行性能分析在模型简单时很有用，但在模型复杂时就不那么有用。

在第五章中，*CUDA 应用程序性能分析和调试*，我们介绍了**NVIDIA 工具扩展**（**NVTX**），它允许我们在 GPU 应用程序中进行自定义注释，并使用 NVIDIA Nsight Systems 查看时间轴。对于复杂的应用程序，程序员分析其性能并找到瓶颈非常有用。

在本节中，我们将介绍如何通过修改 ResNet-50 示例代码在 PyTorch 和 TensorFlow 中使用 NVTX。示例代码可以在本书的 GitHub 存储库的`10_deep_learining/05_framework_profile`文件夹中找到。您可以从[`github.com/nvidia/DeepLearningExamples`](https://github.com/nvidia/DeepLearningExamples)获取原始源代码。

为了简化工作环境配置，我们将使用**NVIDIA GPU 云**（**NGC**）深度学习容器用于 PyTorch 和 TensorFlow。如果您需要了解 NGC 或容器的基本用法，请访问本书附录中的 NGC。

现在，让我们先从 PyTorch 开始。

# 对 PyTorch 模型进行性能分析

在 PyTorch 中，我们可以使用`torch.cuda.nvtx.range_push("foo")`和`torch.cuda.nvtx.range_pop()`来放置自定义标签。这保持了原始的 CUDA NVTX API，即`nvtxRangePush()`和`nvtxRangePop()`。让我们看看 NVTX 注释如何帮助我们在时间轴上理解深度学习操作。在接下来的步骤中，我们将使用`05_framework_profile/pytorch/RN50v1.5`文件中的 ResNet-50 示例代码：

1.  我们将在`train()`函数中的训练循环中放置 NVTX 注释以注释`step`值。该函数可以在`image_classificaiton/training.py`文件中找到。以下截图显示了训练循环和分别在第 234 行和第 260 行的 NVTX 注释：

![](img/a9bbdf88-5cc0-421f-b7ad-00e02655c02c.png)

在上述代码中，训练操作是在`step`函数中实现的，该函数由`get_train_step()`函数定义。因此，我们需要在该函数中放置 NVTX 注释以了解更多信息。

1.  让我们在第 164 行的`get_train_step()`函数中添加一些 NVTX 注释。该函数返回`_step()`函数，其中包括训练操作。因此，我们将在该函数中放置 NVTX 注释。训练过程包括前向和反向传播、全局归约和优化（更新权重）。以下截图显示了在第 166 行和第 171 行的前向传播的注释：

![](img/0c8cae13-132c-4220-afe8-224397161611.png)

通过这种方式，我们可以在其余操作上放置其他注释。

1.  我们还可以为模型层添加 NVTX 注释。在这个例子中，ResNet-50 模型是在`image_classification/resnet.py`文件中实现的。以下截图显示了网络的示例注释：

![](img/c3ac1744-4518-41a9-bfea-8c7cbaf8be42.png)

正如我们所看到的，我们可以按照 ResNet 架构放置 NVTX 注释。如果我们在每个构建块中放置注释，我们可以获得更多信息。

1.  现在，让我们对模型进行分析。正如我们之前讨论的，我们将使用 NGC 深度学习容器，即 PyTorch。`imagenet`数据集位于`/raid/datasets/imagenet/raw-data`文件夹中。为了限制分析时间范围，我们将使用延迟选项（`-y`）和持续时间选项（`-d`）。以下代码显示了一个执行容器并对网络进行分析的 bash shell 脚本：

```cpp
#/bin/bash

CODE_PATH="RN50v1.5"
DATASET_PATH="/raid/datasets/imagenet/raw-data/"
OUTPUT_NAME="resnet50_pyt"

# default profile
docker run --rm -ti --runtime=nvidia \
    -v $(pwd)/${CODE_PATH}:/workspace \
    -v ${DATASET_PATH}:/imagenet \
    nvcr.io/nvidia/pytorch:19.08-py3 \
       nsys profile -t cuda,nvtx,cudnn,cublas -o ${OUTPUT_NAME} 
         -f true -w true -y 60 -d 20 \
       python /workspace/main.py --arch resnet50 -b 64 
         --fp16 /imagenet
```

执行后，上述代码将在 RN50v1.5 目录中生成 profiled 结果，即`resnet50_pyt.qdrep`。

1.  最后，使用 NVIDIA Nsight Systems 打开 profiled 输出`resnet50_pyt.qdrep`，并查看操作。以下截图显示了带有 NVTX 注释的测量步骤：

![](img/2e2912c2-781b-46c0-9e0f-b8c5f97a3f29.png)

在这里，我们可以看到反向操作所花费的时间是前向操作的两倍。此外，PyTorch 将主机线程分开用于训练循环和反向传播。从内核分析来看，耗时最长的点是逐元素的内核执行。让我们扩大前向传递以查看层的执行时间，如下截图所示：

![](img/013b8dd6-8fa4-4942-91b5-0519a77b7f4f.png)

在这里，我们可以看到第二个卷积块需要最长的时间来完成。如果这一层存在效率低下的点，我们可以进一步挖掘。如果某个操作被确定为瓶颈并需要优化，我们还可以使用 NVIDIA Nsight Compute 来分析特定的内核函数。比较主机 API 跟踪和 GPU，我们可以看到时间持续时间是不同的。这是因为主机和 GPU 操作是异步的。因此，当我们从主机测量 GPU 执行时间时，我们需要谨慎。现在，让我们看一下优化步骤，如下截图所示：

![](img/79d227f1-a5aa-4222-88a3-d62710299655.png)

我们可以看到，从主机和 GPU 的测量执行时间中存在巨大差异。主机的测量执行时间为 25.367 毫秒，而 GPU 的时间为 4.048 毫秒。其操作主要是逐元素操作，其执行被延迟直到反向传播完成。我们还可以找到异步执行。之后，我们可以看到`cudaDeviceSynchronize()`函数，该函数防止当前步骤被下一步骤更新。

我们还可以通过设置环境来禁用这些异步操作，即`CUDA_LAUNCH_BLOCKING=1`。我们可以使用环境选项（`-e`）将其传递给 Nsight System 的配置选项。然后，我们可以分析应用程序的`align`操作与主机和内核函数。

PyTorch 在其 CUDA 对象中具有几个具有 NVTX 特色的 API。 PyTorch 文档可以在[`pytorch.org/docs/stable/_modules/torch/cuda/nvtx.html`](https://pytorch.org/docs/stable/_modules/torch/cuda/nvtx.html)找到。通过直接在 PyTorch 中调用 NVTX API，将调用 CUDA NVTX API。这意味着我们可以在分析时间线中获得自定义标记的 NVTX 标记。

# 对 TensorFlow 模型进行分析

对 TensorFlow 图进行分析需要使用启用 NVTX 注释的 NVTX 插件。要在 TensorFlow 中使用 NVTX 注释，我们需要使用以下命令安装`nvtx-plugins-tf` Python 插件：

```cpp
$ pip install nvtx-plugins-tf
```

但是，如果我们使用的是版本晚于 19.08 的 NGC TensorFlow 容器，则无需执行此操作。

TensorFlow 图形 API 是符号 API，因此它们需要特定的编程方法。 NVTX 插件为此提供了两个选项：装饰器和 Python 函数。

以下是 NVTX 装饰器的示例：

```cpp
import nvtx.plugins.tf as nvtx_tf
ENABLE_NVTX=true
@nvtx_tf.ops.trace(message='Dense Block', domain_name='Forward',
        grad_domain_name='Gradient', enabled=ENABLE_NVTX, 
        trainable=True)
def dense_layer(x):
    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, name='dense_1')
    x = tf.layers.dense(x, 1000, activation=tf.nn.relu, name='dense_2’) 
return x
```

以下是 NVTX Python 函数的示例：

```cpp
import nvtx.plugins.tf as nvtx_tf
ENABLE_NVTX=true
x, nvtx_context = nvtx_tf.ops.start(x, message='Dense Block', \ 
        domain_name='Forward’, grad_domain_name='Gradient’, 
        enabled=ENABLE_NVTX, trainable=True)
x = tf.layers.dense(x, 1000, activation=tf.nn.relu, name='dense_1')
x = tf.layers.dense(x, 1000, activation=tf.nn.relu, name='dense_2’) 
x = nvtx_tf.ops.end(x, nvtx_context)
```

NVTX 插件提供了 NVTXHook，它允许我们对 TF 估算器和会话进行分析。例如，我们可以按以下方式使用该钩子：

```cpp
from nvtx.plugins.tf.estimator import NVTXHook

nvtx_callback = NVTXHook(skip_n_steps=1, name='Train’)
training_hooks=[]
training_hooks.append(nvtx_callback)
```

然后，我们可以使用以下代码将其应用于任一选项：

```cpp
with tf.train.MonitoredSession(hooks=training_hooks) as sess:
```

或者，我们可以使用以下代码：

```cpp
tf.estimator.Estimator(hooks=training_hooks, ...)
```

现在，让我们将其应用到示例 ResNet-50 代码中并进行操作审查。示例代码可以在`05_framework_profile/tensorflow/RN50v1.5`文件夹中找到：

1.  让我们首先将`NVTXHook`应用于估算器。训练图的定义可以在`runtime/runner.py`文件的第 312 行找到。在构建图之前，我们将`NVTXHook`附加到钩子列表中，如下面的代码块所示：

![](img/e773aace-88c2-445b-9701-5506ba0bf8d1.png)

1.  然后，我们将 NVTX 注释应用于模型构建函数。`model_build()`函数可以在`model/resnet_v1_5.py`文件的`ResnetModel`类中找到。以下代码显示了如何在`model_build()`函数中的`conv1`层上使用 Python 函数放置 NVTX 注释的示例：

![](img/52fc56c0-b159-4e56-8899-67ab37fb9f4c.png)

在上述代码中，当使用`nvtx_tf.ops.start()`和`nvtx_tf.ops.end()`函数时，我们需要谨慎选择适当的输入和输出。只在其他层中放置 NVTX 注释。确保最终的全连接层输出是网络的输出。

我们还必须禁用用于检查可训练变量数量的代码。如果 NVTX 的`trainable`参数值为`True`，则大小会发生变化。在`resnet_v1_5.py`文件的第 174 行，有一段断言代码，用于检查该变量的数量。只需将其注释掉，如下所示：

![](img/e4654a73-e106-4d11-a41b-c66ccd5d104f.png)

1.  我们还使用 NVTX 装饰器来构建 ResNet 模块。在`model/blocks`目录中，我们可以在`conv2d_blocks.py`和`resnet_bottleneck_block.py`中找到`conv2d`和 ResNet 瓶颈块的实现。在`conv2d_blocks.py`文件中，我们可以装饰`conv2d_block()`函数以注释 NVTX 分析，如下所示：

![](img/abed69dc-571e-4970-af59-ba04f3dd2ac6.png)

同样，我们也可以对`resnet_bottleneck_block.py`文件执行相同操作：

![](img/1edce41d-4992-46c0-b7d6-532b4bfcb157.png)

1.  现在，让我们对模型进行性能分析。就像我们使用 PyTorch 容器一样，我们将使用 TensorFlow 的 NGC 容器。我们假设`imagenet`数据集的`tfrecord`文件位于`/raid/datasets/imagenet/tfrecord`目录中。以下代码显示了一个执行容器并对网络进行性能分析的 bash shell 脚本：

```cpp
#/bin/bash

CODE_PATH="RN50v1.5"
DATASET_PATH="/raid/datasets/imagenet/tfrecord"
OUTPUT_NAME="resnet50_tf"

# default profile
docker run --rm -ti --runtime=nvidia \
    -v $(pwd):/result \
    -v $(pwd)/${CODE_PATH}:/workspace \
    -v ${DATASET_PATH}:/imagenet \
    nvcr.io/nvidia/tensorflow:19.08-py3 \
        nsys profile -t cuda,nvtx,cudnn,cublas -o ${OUTPUT_NAME} 
                     -f true -w true -y 40 -d 20 \
            python /workspace/main.py --mode=training_benchmark 
                                      --warmup_steps 200 \
                --num_iter 500 --iter_unit batch 
                --results_dir=results --batch_size 64
```

当我们执行这个函数时，我们将在`RN50v1.5`目录中得到`resnet50_tf.qdrep`文件。

1.  最后，让我们使用 NVIDIA Nsight System 审查分析输出：

![](img/f5e3118b-c0bc-4cee-83c2-69947821d1f9.png)

在这里，我们可以确认反向传播所花费的时间是前向传播的两倍。这个示例代码与 CPU 和 GPU 不同步。因此，我们可以看到主机和 GPU 之间的时间差异更大。当我们在构建块中放置额外的注释时，我们将能够在层中看到子块的注释。

使用 NVIDIA Nsight Systems 进行性能分析在多 GPU 训练中监视所有归约操作的执行时间时提供了额外的好处。以下截图显示了一个使用两个 GPU 进行训练的 GPU 的性能分析结果：

![](img/e05ad54d-f8a5-482a-935c-f25ed3479b98.png)

在突出显示的行中，我们可以看到`ncclAllRecude()`函数，它同时调用了反向传播。通过这样做，我们不会延迟所有归约操作。这个示例代码使用 Horovod 来训练多个 GPU。如果你想了解更多，请访问 Horovod 的 GitHub 页面：[`github.com/horovod/horovod`](https://github.com/horovod/horovod)。你可以从这里获取文档和示例代码。

# 总结

在本章中，我们学习了如何使用 CUDA 库进行深度学习和性能优势。在回顾它们的用途时，我们将它们与每个步骤的深度学习机制进行匹配。由于我们可以使用的深度学习库，我们可以实现一个简单的 CNN，而不必实现算法。然后，我们使用 NVTX 注释在 PyTorch 和 TensorFlow 中对 ResNet-50 模型进行了性能分析。

对于一些深度学习工程师和研究人员来说，实现基本算法可能是不切实际的。然而，了解性能因素和基本操作可以帮助您构建高效和有效的基于深度学习的产品。如今，我们看到许多产品化的基于深度学习的服务。工程师们花费大量资源将他们训练好的模型产品化，以及训练他们的模型，以便获得尽可能低的错误率。希望您能够了解如何在深度学习应用中使用 NVTX 性能分析。利用这些知识，您可以更好地利用您的 GPU。祝你好运！
