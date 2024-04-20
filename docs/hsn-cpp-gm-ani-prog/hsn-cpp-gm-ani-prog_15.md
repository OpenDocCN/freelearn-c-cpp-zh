# *第十五章*：渲染实例化人群

这最后一章探讨了如何使用实例化来渲染大型人群。人群渲染是一个有趣的话题，因为它将姿势生成（采样）和混合移动到了 GPU 上，使整个动画流水线在顶点着色器中运行。

将姿势生成移动到顶点着色器中，需要将动画信息编码到纹理中。本章的重点将是将动画数据编码到纹理中，并使用该纹理创建动画姿势。

没有实例化，绘制大量人群意味着需要进行大量的绘制调用，这将影响帧率。使用实例化，一个网格可以被多次绘制。如果只有一个绘制调用，人群中每个角色的动画姿势将需要不同的生成。

在本章中，您将探讨将动画采样移动到顶点着色器中以绘制大型人群。本章将涵盖以下主题：

+   在纹理中存储任意数据

+   从纹理中检索任意数据

+   将动画烘焙到纹理中

+   在顶点着色器中对动画纹理进行采样

+   优化人群系统

# 在纹理中存储数据

在 GPU 上进行动画采样并不是一件简单的事情。有很多循环和函数，这使得在 GPU 上进行动画采样成为一个困难的问题。解决这个问题的一种方法是简化它。

与实时采样动画不同，可以在设定的时间间隔内进行采样。在设定的时间间隔内对动画进行采样并将结果数据写入文件的过程称为烘焙。

动画数据烘焙后，着色器就不再需要采样实际的动画片段。相反，它可以根据时间查找最近的采样姿势。那么，这些动画数据烘焙到哪里呢？动画可以烘焙到纹理中。纹理可以用作数据缓冲区，并且已经有一种简单的方法在着色器中读取纹理数据。

通常，纹理中的存储类型和信息都是由着色器中的采样函数抽象出来的。例如，GLSL 中的`texture2D`函数以归一化的`uv`坐标作为参数，并返回一个四分量向量，其值范围从`0`到`1`。

但是纹理中的信息并不是这样的。当使用`glTexImage2D`创建纹理时，它需要一个内部纹理格式（`GL_RGBA`），一个源格式（通常再次是`GL_RGBA`）和一个数据类型（通常是`GL_UNSIGNED_BYTE`）。这些参数用于将底层数据类型转换为`texture2D`返回的归一化值。

在将任意数据存储在纹理中时，存在两个问题。第一个是数据的粒度。在`GL_RGBA`的情况下，每个采样的浮点分量只有 256 个唯一值。第二，如果需要存储的值不是归一化到`0`到`1`范围内的呢？

这就是浮点纹理的用武之地。您可以创建一个具有`GL_RGBA32F`格式的四分量浮点纹理。这个纹理会比其他纹理大得多，因为每个像素将存储四个完整的 32 位浮点数。

浮点纹理可以存储任意数据。在接下来的部分，您将学习如何从浮点纹理中检索任意数据。之后，您将探讨着色器如何从浮点纹理中读取数据。

# 从纹理中读取数据

本节探讨了如何在着色器中检索存储在纹理中的动画数据。在本节中，您将学习如何对纹理进行采样以及在采样纹理时应该使用哪些采样器状态。

一旦数据格式正确，对其进行采样就成为下一个挑战。`glTexImage2D`函数期望归一化的`uv`坐标并返回一个归一化值。另一方面，`texelFetch`函数可以用于使用像素坐标对纹理进行采样并返回这些坐标处的原始数据。

`texelFetch` glsl 接受三个参数：一个采样器，一个`ivec2`和一个整数。`ivec2`是被采样的像素的*x*和*y*坐标，以像素空间为单位。最后一个整数是要使用的 mip 级别，对于本章来说，将始终为`0`。

mipmap 是同一图像的逐渐降低分辨率版本的链。当 mip 级别缩小时，数据会丢失。这种数据丢失会改变动画的内容。避免为动画纹理生成 mip。

因为需要以与写出时完全相同的方式读取数据，任何插值也会破坏动画数据。确保使用最近邻采样来对动画纹理进行采样。

使用`texelFetch`而不是`glTexImage2D`来对纹理进行采样应该返回正确的数据。纹理可以在顶点着色器或片段着色器中进行采样。在下一节中，您将探索这些浮点纹理中应该存储什么动画数据。

# 编码动画数据

现在你知道如何读取和写入数据到纹理了，下一个问题是，纹理中需要写入什么数据？你将把动画数据编码到纹理中。每个动画片段将在设定的间隔内进行采样。所有这些样本的结果姿势将存储在纹理中。

为了编码这些数据，纹理的*x*轴将表示时间。纹理的*y*轴将表示正在进行动画的骨骼。每个骨骼将占用三行：一个用于位置，一个用于旋转，一个用于缩放。

动画片段将在设定的间隔内进行采样，以确保纹理的宽度有多少个样本。例如，对于一个*256x256*的动画纹理，动画片段将需要被采样 256 次。

在对动画片段进行采样以将其编码到纹理中时，对于每个样本，您将找到每个骨骼的世界空间变换并将其写入纹理。*y*坐标将是`joint_index * 3 + component`，其中有效的组件是`position = 0`，`rotation = 1`和`scale = 3`。

一旦这些值被写入纹理，就将纹理上传到 GPU 并使用它。在下一节中，您将探索着色器如何评估这个动画纹理。

# 探索每个实例数据

在渲染大量人群时，人群中的每个演员都有特定的属性。在本节中，您将探索每个实例数据是什么，以及如何将其传递给着色器。这将大大减少每帧上传到 GPU 的统一数组的数据量。

将蒙皮管道移动到顶点着色器并不能完全消除需要将与人群相关的统一数据传递给着色器。人群中的每个演员都需要一些数据上传到 GPU。每个实例数据比使用姿势调色板矩阵上传的数据要小得多。

人群中的每个演员都需要位置、旋转和缩放来构建模型矩阵。演员需要知道当前帧进行采样以及当前帧和下一帧之间的时间来进行混合。

每个演员实例数据的总大小是 11 个浮点数和 2 个整数。每个实例只有 52 个字节。每个实例数据将始终使用统一数组传递。数组的大小是人群包含的演员数量。数组的每个元素代表一个独特的演员。

着色器将负责从每个实例数据和动画纹理构建适当的矩阵。当前帧和下一帧之间的混合是可选的；混合可能不会 100%正确，但它应该看起来还不错。

在下一节中，您将实现一个`AnimationTexture`类，它将让您在代码中使用动画纹理。

# 创建动画纹理

在这一节中，您将实现所有需要在`AnimTexture`类中使用浮点纹理的代码。每个`AnimTexture`对象将包含一个 32 位浮点 RGBA 纹理。这些数据将有两份：一份在 CPU 上，一份上传到 GPU 上。

CPU 缓冲区保留下来，以便在保存到磁盘之前或上传到 OpenGL 之前轻松修改纹理的内容。这样做可以简化 API，但会增加一些额外的内存。

没有标准的 32 位纹理格式，因此保存和写入磁盘将简单地将`AnimTexture`类的二进制内容转储到磁盘上。在下一节中，您将开始实现`AnimTexture`类。这个类将提供一个易于使用的接口，用于实现 32 位浮点纹理。

## 声明 AnimTexture 类

动画纹理被假定总是正方形的；宽度和高度不需要分别跟踪。使用单个大小变量应该足够了。`AnimTexture`类将始终在内存中同时拥有两份纹理，一份在 CPU 上，一份在 GPU 上。

创建一个名为`AnimTexture.h`的新文件，并在这个文件中声明`AnimTexture`类。按照以下步骤声明`AnimTexture`类：

1.  声明`AnimTexture`类。它有三个成员变量：一个浮点数组，一个纹理大小的整数，以及一个指向 OpenGL 纹理对象的句柄：

```cpp
    class AnimTexture {
    protected:
        float* mData;
        unsigned int mSize;
        unsigned int mHandle;
    ```

1.  声明`AnimTexture`具有默认构造函数、复制构造函数、赋值运算符和析构函数：

```cpp
    public:
        AnimTexture();
        AnimTexture(const AnimTexture&);
        AnimTexture& operator=(const AnimTexture&);
        ~AnimTexture();
    ```

1.  声明函数，以便将`AnimTexture`保存到磁盘并再次加载：

```cpp
        void Load(const char* path);
        void Save(const char* path);
    ```

1.  声明一个函数，将数据从`mData`变量上传到 OpenGL 纹理：

```cpp
        void UploadTextureDataToGPU();
    ```

1.  声明`AnimTexture`包含的 CPU 端数据的 getter 和 setter 函数：

```cpp
        unsigned int Size();
        void Resize(unsigned int newSize);
        float* GetData();
    ```

1.  声明`GetTexel`，它接受*x*和*y*坐标并返回一个`vec4`，以及一个`SetTexel`函数来设置`vec3`或`quat`对象。这些函数将写入纹理的数据：

```cpp
        void SetTexel(unsigned int x, unsigned int y, 
                      const vec3& v);
        void SetTexel(unsigned int x, unsigned int y, 
                      const quat& q);
        vec4 GetTexel(unsigned int x, unsigned int y);
    ```

1.  声明绑定和解绑纹理以进行渲染的函数。这将与`Texture`类的`Set`和`Unset`函数的方式相同：

```cpp
       void Set(unsigned int uniform, unsigned int texture);
       void UnSet(unsigned int textureIndex);
       unsigned int GetHandle();
    };
    ```

`AnimTexture`类是一种方便的处理浮点纹理的方式。`get`和`SetTexel`方法可以使用直观的 API 读取和写入纹理。在下一节中，您将开始实现`AnimTexture`类。

## 实现`AnimTexture`类

在这一节中，您将实现`AnimTexture`类，其中包含用于处理浮点纹理的 OpenGL 代码，并提供一个易于使用的 API。如果您想使用除了 OpenGL 之外的图形 API，那么这个类将需要使用该 API 进行重写。

当`AnimTexture`保存到磁盘时，整个`mData`数组将作为一个大的二进制块写入文件。这个大的纹理数据占用了相当多的内存；例如，一个*512x512*的纹理大约占用 4MB。纹理压缩不适用，因为动画数据需要精确。

`SetTexel`函数是我们将要写入动画纹理数据的主要方式。这些函数接受*x*和*y*坐标，以及`vec3`或四元数值。函数需要根据给定的*x*和*y*坐标找出`mData`数组中的正确索引，然后相应地设置像素值。

创建一个名为`AnimTexture.cpp`的新文件。在这个新文件中实现`AnimTexture`类。现在，按照以下步骤实现`AnimTexture`类：

1.  实现默认构造函数。它应该将数据和大小设置为零，并生成一个新的 OpenGL 着色器句柄：

```cpp
    AnimTexture::AnimTexture() {
        mData = 0;
        mSize = 0;
        glGenTextures(1, &mHandle);
    }
    ```

1.  实现复制构造函数。它应该做与默认构造函数相同的事情，并使用赋值运算符来复制实际的纹理数据：

```cpp
    AnimTexture::AnimTexture(const AnimTexture& other) {
        mData = 0;
        mSize = 0;
        glGenTextures(1, &mHandle);
        *this = other;
    }
    ```

1.  实现赋值运算符。它只需要复制 CPU 端的数据；OpenGL 句柄可以不变：

```cpp
    AnimTexture& AnimTexture::operator=(
                              const AnimTexture& other) {
        if (this == &other) {
            return *this;
        }
        mSize = other.mSize;
        if (mData != 0) {
            delete[] mData;
        }
        mData = 0;
        if (mSize != 0) {
            mData = new float[mSize * mSize * 4];
            memcpy(mData, other.mData, 
                sizeof(float) * (mSize * mSize * 4));
        }
        return *this;
    }
    ```

1.  实现`AnimTexture`类的析构函数。它应该删除内部浮点数组，并释放类所持有的 OpenGL 句柄：

```cpp
    AnimTexture::~AnimTexture() {
        if (mData != 0) {
            delete[] mData;
        }
        glDeleteTextures(1, &mHandle);
    }
    ```

1.  实现`Save`函数。它应该将`AnimTexture`的大小写入文件，并将`mData`的内容作为一个大的二进制块写入：

```cpp
    void AnimTexture::Save(const char* path) {
        std::ofstream file;
        file.open(path, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            cout << "Couldn't open " << path << "\n";
        }
        file << mSize;
        if (mSize != 0) {
            file.write((char*)mData, 
                 sizeof(float) * (mSize * mSize * 4));
        }
        file.close();
    }
    ```

1.  实现`Load`函数，将序列化的动画数据加载回内存：

```cpp
    void AnimTexture::Load(const char* path) {
        std::ifstream file;
        file.open(path, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            cout << "Couldn't open " << path << "\n";
        }
        file >> mSize;
        mData = new float[mSize * mSize * 4];
        file.read((char*)mData, 
             sizeof(float) * (mSize * mSize * 4));
        file.close();
        UploadTextureDataToGPU();
    }
    ```

1.  实现`UploadDataToGPU`函数。它的实现方式与`Texture::Load`非常相似，但使用的是`GL_RGBA32F`而不是`GL_FLOAT`：

```cpp
    void AnimTexture::UploadTextureDataToGPU() {
        glBindTexture(GL_TEXTURE_2D, mHandle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mSize, 
                      mSize, 0, GL_RGBA, GL_FLOAT, mData);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 
                        GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 
                        GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, 
                        GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, 
                        GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    ```

1.  实现大小、OpenGL 句柄和浮点数据获取函数：

```cpp
    unsigned int AnimTexture::Size() {
        return mSize;
    }
    unsigned int AnimTexture::GetHandle() {
        return mHandle;
    }
    float* AnimTexture::GetData() {
        return mData;
    }
    ```

1.  实现`resize`函数，它应该设置`mData`数组的大小。这个函数的参数是动画纹理的宽度或高度：

```cpp
    void AnimTexture::Resize(unsigned int newSize) {
        if (mData != 0) {
            delete[] mData;
        }
        mSize = newSize;
        mData = new float[mSize * mSize * 4];
    }
    ```

1.  实现`Set`函数。它的工作方式类似于`Texture::Set`：

```cpp
    void AnimTexture::Set(unsigned int uniformIndex, unsigned int textureIndex) {
        glActiveTexture(GL_TEXTURE0 + textureIndex);
        glBindTexture(GL_TEXTURE_2D, mHandle);
        glUniform1i(uniformIndex, textureIndex);
    }
    ```

1.  实现`UnSet`函数。它的工作方式类似于`Texture::UnSet`：

```cpp
    void AnimTexture::UnSet(unsigned int textureIndex) {
        glActiveTexture(GL_TEXTURE0 + textureIndex);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
    }
    ```

1.  实现`SetTexel`函数，它以矢量`3`作为参数。这个函数应该将像素的未使用的 A 分量设置为`0`：

```cpp
    void AnimTexture::SetTexel(unsigned int x, 
                      unsigned int y, const vec3& v) {
        unsigned int index = (y * mSize * 4) + (x * 4);
        mData[index + 0] = v.x;
        mData[index + 1] = v.y;
        mData[index + 2] = v.z;
        mData[index + 3] = 0.0f;
    }
    ```

1.  实现`SetTexel`函数，它以四元数作为参数：

```cpp
    void AnimTexture::SetTexel(unsigned int x, 
                      unsigned int y, const quat& q) {
        unsigned int index = (y * mSize * 4) + (x * 4);
        mData[index + 0] = q.x;
        mData[index + 1] = q.y;
        mData[index + 2] = q.z;
        mData[index + 3] = q.w;
    }
    ```

1.  实现`GetTexel`函数。这个函数将始终返回一个`vec4`，其中包含像素的每个分量：

```cpp
    vec4 AnimTexture::GetTexel(unsigned int x, 
                               unsigned int y) {
        unsigned int index = (y * mSize * 4) + (x * 4);
        return vec4(
            mData[index + 0],
            mData[index + 1],
            mData[index + 2],
            mData[index + 3]
        );
    }
    ```

在本节中，您学会了如何创建一个 32 位浮点纹理并管理其中的数据。`AnimTexture`类应该让您使用直观的 API 来处理浮点纹理，而不必担心任何 OpenGL 函数。在下一节中，您将创建一个函数，该函数将对动画剪辑进行采样，并将结果的动画数据写入纹理。

# 动画烘焙器

在本节中，您将学习如何将动画剪辑编码到动画纹理中。这个过程称为烘焙。

使用一个辅助函数实现纹理烘焙。这个`Bake`函数将在设定的间隔内对动画进行采样，并将每个采样的骨骼层次结构写入浮点纹理中。

对于参数，`Bake`函数需要一个骨架、一个动画剪辑，以及一个要写入的`AnimTexture`的引用。骨架很重要，因为它提供了静止姿势，这将用于动画剪辑中不存在的任何关节。骨架的每个关节都将被烘焙到纹理中。让我们开始吧：

1.  创建一个名为`AnimBaker.h`的新文件，并在其中添加`BakeAnimationToTexture`函数的声明：

```cpp
    void BakeAnimationToTexture(Skeleton& skel, Clip& clip, 
                                AnimTexture& outTex);
    ```

1.  创建一个名为`AnimBaker.cpp`的新文件。开始在这个文件中实现`BakeAnimationToTexture`函数：

```cpp
    void BakeAnimationToTexture(Skeleton& skel, Clip& clip, 
                                AnimTexture& tex) {
        Pose& bindPose = skel.GetBindPose();
    ```

1.  要将动画烘焙到纹理中，首先创建一个动画将被采样到的姿势。然后，循环遍历纹理的*x*维度，即时间：

```cpp
        Pose pose = bindPose;
        unsigned int texWidth = tex.Size();
        for (unsigned int x = 0; x < texWidth; ++x) {
    ```

1.  对于每次迭代，找到迭代器的归一化值（迭代器索引/（大小-1））。将归一化时间乘以剪辑的持续时间，然后加上剪辑的开始时间。在当前像素的这个时间点对剪辑进行采样：

```cpp
            float t = (float)x / (float)(texWidth - 1);
            float start = clip.GetStartTime();
            float time = start + clip.GetDuration() * t;
            clip.Sample(pose, time);
    ```

1.  一旦剪辑被采样，就循环遍历绑定姿势中的所有关节。找到当前关节的全局变换，并使用`SetTexel`将数据写入纹理：

```cpp
            for (unsigned int y = 0;y<pose.Size()*3;y+=3) {
               Transform node=pose.GetGlobalTransform(y/3);
               tex.SetTexel(x, y + 0, node.position);
               tex.SetTexel(x, y + 1, node.rotation);
               tex.SetTexel(x, y + 2, node.scale);
            }
    ```

1.  在`Bake`函数返回之前，调用提供的动画纹理上的`UploadTextureDataToGPU`函数。这将使纹理在被烘焙后立即可用：

```cpp
        } // End of x loop
        tex.UploadTextureDataToGPU();
    }
    ```

在高层次上，动画纹理被用作时间轴，其中*x*轴是时间，*y*轴是该时间点上动画关节的变换。在下一节中，您将创建人群着色器。人群着色器使用`BakeAnimationToTexture`烘焙到纹理中的数据来采样动画的当前姿势。

# 创建人群着色器

要呈现一个群众，您需要创建一个新的着色器。群众着色器将具有投影和视图统一，但没有模型统一。这是因为所有演员都是用相同的投影和视图矩阵绘制的，但需要一个独特的模型矩阵。着色器将有三个统一数组：一个用于位置，一个用于旋转，一个用于比例，而不是模型矩阵。

将放入这些数组的值是一个实例索引-当前正在呈现的网格的索引。每个顶点都通过内置的`glsl`变量`gl_InstanceID`获得其网格实例的副本。每个顶点将使用位置、旋转和比例统一数组构造一个模型矩阵。

反向绑定姿势就像一个矩阵统一数组，具有常规的蒙皮，但动画姿势不是。要找到动画姿势，着色器将不得不对动画纹理进行采样。由于每个顶点被绑定到四个顶点，所以必须为每个顶点找到四次动画姿势。

创建一个名为`crowd.vert`的新文件。群众着色器将在此文件中实现。按照以下步骤实现群众着色器：

1.  通过定义两个常量来开始实现着色器：一个用于骨骼的最大数量，一个用于支持的实例的最大数量：

```cpp
    #version 330 core
    #define MAX_BONES 60
    #define MAX_INSTANCES 80
    ```

1.  声明所有群众演员共享的制服。这包括视图和投影矩阵，反向绑定姿势调色板和动画纹理：

```cpp
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat4 invBindPose[MAX_BONES];
    uniform sampler2D animTex;
    ```

1.  声明每个群众演员独有的统一。这包括演员的变换，当前和下一帧，以及混合时间：

```cpp
    uniform vec3 model_pos[MAX_INSTANCES];
    uniform vec4 model_rot[MAX_INSTANCES];
    uniform vec3 model_scl[MAX_INSTANCES];
    uniform ivec2 frames[MAX_INSTANCES];
    uniform float time[MAX_INSTANCES];
    ```

1.  声明顶点结构。每个顶点的数据与任何蒙皮网格的数据相同：

```cpp
    in vec3 position;
    in vec3 normal;
    in vec2 texCoord;
    in vec4 weights;
    in ivec4 joints;
    ```

1.  声明群众着色器的输出值：

```cpp
    out vec3 norm;
    out vec3 fragPos;
    out vec2 uv;
    ```

1.  实现一个函数，该函数将一个向量和一个四元数相乘。这个函数将与您在[*第四章*]（B16191_04_Final_JC_ePub.xhtml#_idTextAnchor069）*实现四元数*中构建的`transformVector`函数具有相同的实现，只是它在着色器中运行：

```cpp
    vec3 QMulV(vec4 q, vec3 v) {
        return q.xyz * 2.0f * dot(q.xyz, v) +
               v * (q.w * q.w - dot(q.xyz, q.xyz)) +
               cross(q.xyz, v) * 2.0f * q.w;
    }
    ```

1.  实现`GetModel`函数。给定一个实例索引，该函数应该从动画纹理中采样并返回一个*4x4*变换矩阵：

```cpp
    mat4 GetModel(int instance) {
        vec3 position = model_pos[instance];
        vec4 rotation = model_rot[instance];
        vec3 scale = model_scl[instance];
        vec3 xBasis = QMulV(rotation, vec3(scale.x, 0, 0));
        vec3 yBasis = QMulV(rotation, vec3(0, scale.y, 0));
        vec3 zBasis = QMulV(rotation, vec3(0, 0, scale.z));
        return mat4(
            xBasis.x, xBasis.y, xBasis.z, 0.0,
            yBasis.x, yBasis.y, yBasis.z, 0.0,
            zBasis.x, zBasis.y, zBasis.z, 0.0,
            position.x, position.y, position.z, 1.0
        );
    }
    ```

1.  使用关节和实例实现`GetPose`函数，该函数应返回关节的动画世界矩阵。通过找到 x 和 y 位置来采样动画纹理开始实现：

```cpp
    mat4 GetPose(int joint, int instance) {
        int x_now = frames[instance].x;
        int x_next = frames[instance].y;
        int y_pos = joint * 3;
    ```

1.  从动画纹理中采样当前帧的位置、旋转和比例：

```cpp
        vec4 pos0 = texelFetch(animTex, ivec2(x_now, 
                              (y_pos + 0)), 0);
        vec4 rot0 = texelFetch(animTex, ivec2(x_now, 
                              (y_pos + 1)), 0);
        vec4 scl0 = texelFetch(animTex, ivec2(x_now, 
                              (y_pos + 2)), 0);
    ```

1.  从动画纹理中采样下一帧的位置、旋转和比例：

```cpp
        vec4 pos1 = texelFetch(animTex, ivec2(x_next, 
                              (y_pos + 0)), 0);
        vec4 rot1 = texelFetch(animTex, ivec2(x_next, 
                              (y_pos + 1)), 0);
        vec4 scl1 = texelFetch(animTex, ivec2(x_next, 
                              (y_pos + 2)), 0);
    ```

1.  在两个帧之间进行插值：

```cpp
        if (dot(rot0, rot1) < 0.0) { rot1 *= -1.0; }
        vec4 position = mix(pos0, pos1, time[instance]);
        vec4 rotation = normalize(mix(rot0, 
                                  rot1, time[instance]));
        vec4 scale = mix(scl0, scl1, time[instance]);
    ```

1.  使用插值的位置、旋转和比例返回一个 4x4 矩阵：

```cpp
        vec3 xBasis = QMulV(rotation, vec3(scale.x, 0, 0));
        vec3 yBasis = QMulV(rotation, vec3(0, scale.y, 0));
        vec3 zBasis = QMulV(rotation, vec3(0, 0, scale.z));
        return mat4(
            xBasis.x, xBasis.y, xBasis.z, 0.0,
            yBasis.x, yBasis.y, yBasis.z, 0.0,
            zBasis.x, zBasis.y, zBasis.z, 0.0,
            position.x, position.y, position.z, 1.0
        );
    }
    ```

1.  通过找到着色器的主函数来实现着色器的主要功能，找到所有四个动画姿势矩阵，以及群众中当前演员的模型矩阵。使用`gl_InstanceID`来获取当前绘制的演员的 ID：

```cpp
    void main() {
        mat4 pose0 = GetPose(joints.x, gl_InstanceID);
        mat4 pose1 = GetPose(joints.y, gl_InstanceID);
        mat4 pose2 = GetPose(joints.z, gl_InstanceID);
        mat4 pose3 = GetPose(joints.w, gl_InstanceID);
        mat4 model = GetModel(gl_InstanceID);
    ```

1.  通过找到顶点的`skin`矩阵来继续实现主函数：

```cpp
        mat4 skin = (pose0*invBindPose[joints.x])*weights.x;
        skin += (pose1 * invBindPose[joints.y]) * weights.y;
        skin += (pose2 * invBindPose[joints.z]) * weights.z;
        skin += (pose3 * invBindPose[joints.w]) * weights.w;
    ```

1.  通过将位置和法线通过蒙皮顶点的变换管道来完成实现主函数：

```cpp
        gl_Position = projection * view * model * 
                      skin * vec4(position, 1.0);
        fragPos = vec3(model * skin * vec4(position, 1.0));
        norm = vec3(model * skin * vec4(normal, 0.0f));
        uv = texCoord;
    }
    ```

在本节中，您实现了群众着色器。这个顶点着色器使用动画纹理来构建正在呈现的每个顶点的动画姿势。它将蒙皮管道的姿势生成部分移动到了 GPU 上。该着色器旨在呈现实例化的网格；它使用`gl_InstanceID`来确定当前正在呈现的实例。

这个着色器是一个很好的起点，但总有改进的空间。该着色器目前使用了大量的统一索引。一些低端机器可能提供不了足够的统一。本章末尾将介绍几种优化策略。在下一节中，您将实现一个`Crowd`类来帮助管理 Crowd 着色器需要的所有数据。

# 创建 Crowd 实用程序类

在这一部分，您将构建`Crowd`类。这是一个实用类，可以使用易于使用的 API 渲染大量人群。`Crowd`类封装了人群的状态。

`Crowd`类必须维护类中每个演员的实例数据。为了适应这一点，您需要声明一个最大演员数量。然后，所有特定于演员的信息可以存储在结构数组中，其中索引是演员 ID。

特定于演员的数据包括演员的世界变换，以及与其动画播放相关的数据。动画数据是哪些帧正在插值，插值值，以及当前和下一帧的关键时间。

创建一个名为`Crowd.h`的新文件。`Crowd`类将在此文件中声明。按照以下步骤声明`Crowd`类：

1.  将人群演员的最大数量定义为`80`：

```cpp
    #define CROWD_MAX_ACTORS 80
    ```

1.  通过为所有实例数据创建向量来声明`Crowd`类。这包括每个演员的变换、动画帧和时间的数据，以及帧插值信息：

```cpp
    struct Crowd {
    protected:
        std::vector<vec3> mPositions;
        std::vector<quat> mRotations;
        std::vector<vec3> mScales;
        std::vector<ivec2> mFrames;
        std::vector<float> mTimes;
        std::vector<float> mCurrentPlayTimes;
        std::vector<float> mNextPlayTimes;
    ```

1.  声明`AdjustTime`、`UpdatePlaybackTimes`、`UpdateFrameIndices`和`UpdateInterpolationTimes`函数。`AdjustTime`函数类似于`Clip::AdjustTimeToFitRange`；它确保给定时间是有效的：

```cpp
    protected:
        float AdjustTime(float t, float start, 
                    float end, bool looping);
        void UpdatePlaybackTimes(float dt, bool looping, 
                    float start, float end);
        void UpdateFrameIndices(float start, 
                    float duration, unsigned int texWidth);
        void UpdateInterpolationTimes(float start, 
                    float duration, unsigned int texWidth);
    ```

1.  为人群的大小和每个演员的`Transform`属性声明 getter 和 setter 函数：

```cpp
    public:
        unsigned int Size();
        void Resize(unsigned int size);
        Transform GetActor(unsigned int index);
        void SetActor(unsigned int index, 
                      const Transform& t);
    ```

1.  最后，声明`Update`和`SetUniforms`函数。这些函数将推进当前动画并更新每个实例的着色器 uniforms：

```cpp
        void Update(float deltaTime, Clip& mClip, 
                    unsigned int texWidth);
        void SetUniforms(Shader* shader);
    };
    ```

`Crowd`类为管理人群中每个演员的每个实例信息提供了直观的接口。在下一节中，您将开始实现`Crowd`类。

## 实现 Crowd 类

`Crowd`类为您提供了一种方便的方式来管理人群中的所有演员。这个类的大部分复杂性在于计算正确的播放信息。这项工作在`Update`函数中完成。`Update`函数使用三个辅助函数，即`UpdatePlaybackTimes`、`UpdateFrameIndices`和`UpdateInterpolateionTimes`来工作。

人群中每个演员的当前动画播放时间将存储在`mCurrentPlayTimes`向量中。`mNextPlayTimes`向量是动画的预计下一个时间，这允许两个采样帧进行插值。`UpdatePlaybackTimes`函数将更新这两个向量。

猜测下一帧的播放时间很重要，因为动画纹理的采样率是未知的。例如，如果动画以 240 FPS 编码，并以 60 FPS 播放，那么下一帧将相隔四个采样。

`mFrames`向量包含两个组件整数向量。第一个组件是当前动画帧的`u`纹理坐标。第二个组件是下一帧中将显示的动画帧的`v`纹理坐标。`v`纹理坐标是关节索引。

`UpdateFrameIndex`函数负责更新这个向量。要找到当前帧的*x*坐标，需要对帧时间进行归一化，然后将归一化的帧时间乘以纹理的大小。可以通过从开始时间减去帧时间并将结果除以剪辑的持续时间来归一化帧的时间。

着色器需要在当前动画姿势和下一个动画姿势之间进行插值。为此，它需要知道两个姿势帧之间的当前归一化时间。这存储在`mTimes`变量中。

`mTimes`变量由`UpdateInterpolationTimes`函数更新。该函数找到当前帧的持续时间，然后将播放时间相对于当前帧归一化到该持续时间。

要更新`Crowd`类，您必须按顺序调用`UpdatePlaybackTimes`、`UpdateFrameIndices`和`UpdateInterpolateionTimes`函数。完成后，`Crowd`类可以使用`SetUniforms`函数设置其 uniform 值。

创建一个名为`Crowd.cpp`的新文件。`Crowd`类将在此文件中实现。按照以下步骤实现`Crowd`类：

1.  实现大小的获取器和设置器函数。设置器函数需要设置`Crowd`类中包含的所有向量的`size`：

```cpp
    unsigned int Crowd::Size() {
        return mCurrentPlayTimes.size();
    }
    void Crowd::Resize(unsigned int size) {
        if (size > CROWD_MAX_ACTORS) {
            size = CROWD_MAX_ACTORS;
        }
        mPositions.resize(size);
        mRotations.resize(size);
        mScales.resize(size, vec3(1, 1, 1));
        mFrames.resize(size);
        mTimes.resize(size);
        mCurrentPlayTimes.resize(size);
        mNextPlayTimes.resize(size);
    }
    ```

1.  实现演员变换的获取器和设置器函数。位置、旋转和缩放保存在单独的向量中；演员的获取器和设置器函数隐藏了该实现，而是使用`Transform`对象：

```cpp
    Transform Crowd::GetActor(unsigned int index) {
        return Transform(
            mPositions[index],
            mRotations[index],
            mScales[index] );
    }
    void Crowd::SetActor(unsigned int index, 
                         const Transform& t) {
        mPositions[index] = t.position;
        mRotations[index] = t.rotation;
        mScales[index] = t.scale;
    }
    ```

1.  实现`AdjustTime`函数；它类似于`Clip::AdjustTimeToFitRange`函数：

```cpp
    float Crowd::AdjustTime(float time, float start, 
                            float end, bool looping) {
        if (looping) {
            time = fmodf(time - start, end - start);
            if (time < 0.0f) {
                time += end - start;
            }
            time = time + start;
        }
        else {
            if (time < start) { time = start; }
            if (time > end) { time = end; }
        }
        return time;
    }
    ```

1.  实现`UpdatePlaybackTimes`辅助函数。该函数将按照增量时间推进所有演员的播放时间：

```cpp
    void Crowd::UpdatePlaybackTimes(float deltaTime, 
                bool looping, float start, float end) {
        unsigned int size = mCurrentPlayTimes.size();
        for (unsigned int i = 0; i < size; ++i) {
            float time = mCurrentPlayTimes[i] + deltaTime;
            mCurrentPlayTimes[i] = AdjustTime(time, start,
                                            end, looping);
            time = mCurrentPlayTimes[i] + deltaTime;
            mNextPlayTimes[i] = AdjustTime(time, start, 
                                          end, looping);
        }
    }
    ```

1.  实现`UpdateFrameIndices`函数。该函数将当前播放时间转换为沿动画纹理*x*轴的像素坐标：

```cpp
    void Crowd::UpdateFrameIndices(float start, float duration, unsigned int texWidth) {
        unsigned int size = mCurrentPlayTimes.size();
        for (unsigned int i = 0; i < size; ++i) {
            float thisNormalizedTime = 
                 (mCurrentPlayTimes[i] - start) / duration;
            unsigned int thisFrame = 
                 thisNormalizedTime * (texWidth - 1);
            float nextNormalizedTime = 
                 (mNextPlayTimes[i] - start) / duration;
            unsigned int nextFrame = 
                 nextNormalizedTime * (texWidth - 1);
            mFrames[i].x = thisFrame;
            mFrames[i].y = nextFrame;
        }
    }
    ```

1.  实现`UpdateInterpolationTimes`函数。该函数应该找到当前和下一个动画帧之间的插值时间：

```cpp
    void Crowd::UpdateInterpolationTimes(float start, 
              float duration, unsigned int texWidth) {
        unsigned int size =  mCurrentPlayTimes.size();
        for (unsigned int i = 0; i < size; ++i) {
            if (mFrames[i].x == mFrames[i].y) {
                mTimes[i] = 1.0f;
                continue;
            }
            float thisT = (float)mFrames[i].x / 
                          (float)(texWidth - 1);
            float thisTime = start + duration * thisT;
            float nextT = (float)mFrames[i].y / 
                          (float)(texWidth - 1);
            float nextTime = start + duration * nextT;
            if (nextTime < thisTime) {
                nextTime += duration;
            }
            float frameDuration = nextTime - thisTime;
            mTimes[i] = (mCurrentPlayTimes[i] - thisTime) /
                        frameDuration;
        }
    }
    ```

1.  实现`Update`方法。该方法依赖于`UpdatePlaybackTimes`、`UpdateFrameIndices`和`UpdateInterpolationTimes`辅助函数：

```cpp
    void Crowd::Update(float deltaTime, Clip& mClip, 
                            unsigned int texWidth) {
       bool looping = mClip.GetLooping();
       float start = mClip.GetStartTime();
       float end = mClip.GetEndTime();
       float duration = mClip.GetDuration();

       UpdatePlaybackTimes(deltaTime, looping, start, end);
       UpdateFrameIndices(start, duration, texWidth);
       UpdateInterpolationTimes(start, duration, texWidth);
    }
    ```

1.  实现`SetUniforms`函数，将`Crowd`类中包含的向量传递给人群着色器作为 uniform 数组：

```cpp
    void Crowd::SetUniforms(Shader* shader) {
        Uniform<vec3>::Set(shader->GetUniform("model_pos"),
                           mPositions);
        Uniform<quat>::Set(shader->GetUniform("model_rot"), 
                           mRotations);
        Uniform<vec3>::Set(shader->GetUniform("model_scl"), 
                           mScales);
        Uniform<ivec2>::Set(shader->GetUniform("frames"), 
                           mFrames);
        Uniform<float>::Set(shader->GetUniform("time"), 
                           mTimes);
    }
    ```

使用`Crowd`类应该是直观的：创建一个人群，设置其演员的播放时间和模型变换，然后绘制人群。在下一节中，您将探讨如何使用`Crowd`类来绘制大型人群的示例。

## 使用 Crowd 类

使用`Crowd`类应该是直观的，但渲染代码可能不会立即显而易见。人群着色器的非实例 uniform，如视图或投影矩阵，仍然需要手动设置。`Crowd`类的`Set`函数设置的唯一 uniform 是每个演员的 uniform。

不要使用`Mesh`类的`Draw`方法进行渲染，而是使用`DrawInstanced`方法。对于实例数量参数，传递人群的大小。以下代码片段显示了如何绘制人群的最小示例：

```cpp
void Render(float aspect) {
    mat4 projection = perspective(60.0f, aspect, 0.01f, 100);
    mat4 view=lookAt(vec3(0,15,40), vec3(0,3,0), vec3(0,1,0));
    mCrowdShader->Bind();
    int viewUniform = mCrowdShader->GetUniform("view")
    Uniform<mat4>::Set(viewUniform, view);
    int projUniform = mCrowdShader->GetUniform("projection")
    Uniform<mat4>::Set(projUniform, projection);
    int lightUniform = mCrowdShader->GetUniform("light");
    Uniform<vec3>::Set(lightUniform, vec3(1, 1, 1));
    int invBind = mCrowdShader->GetUniform("invBindPose");
    Uniform<mat4>::Set(invBind, mSkeleton.GetInvBindPose());
    int texUniform = mCrowdShader->GetUniform("tex0");
    mDiffuseTexture->Set(texUniform, 0);
    int animTexUniform = mCrowdShader->GetUniform("animTex");
    mCrowdTexture->Set(animTexUniform, 1);
    mCrowd.SetUniforms(mCrowdShader);
    int pAttrib = mCrowdShader->GetAttribute("position");
    int nAttrib = mCrowdShader->GetAttribute("normal");
    int tAttrib = mCrowdShader->GetAttribute("texCoord");
    int wAttrib = mCrowdShader->GetAttribute("weights");
    int jAttrib = mCrowdShader->GetAttribute("joints");
    mMesh.Bind(pAttrib, nAttrib, uAttrib, wAttrib, jAttrib);
    mMesh.DrawInstanced(mCrowd.Size());
    mMesh.UnBind(pAttrib, nAttrib, uAttrib, wAttrib, jAttrib);
    mCrowdTexture->UnSet(1);
    mDiffuseTexture->UnSet(0);
    mCrowdShader->UnBind();
}
```

在大多数情况下，代码看起来与常规蒙皮网格相似。这是因为`Crowd`类的`SetUniforms`函数设置了特定实例的 uniform 值。其他 uniform 的设置方式与以前相同。在下一节中，您将探讨如何在顶点着色器中混合两个动画。

在本节中，您创建了一个`Crowd`类，它提供了一个易于使用的接口，以便您可以设置`Crowd`着色器所需的 uniform。还介绍了如何使用`Crowd`类来渲染大型人群的演示。

# 混合动画

在顶点着色器中可以在两个动画之间进行混合。有两个原因可能会导致你不希望在顶点着色器中进行动画混合。首先，这样做会使着色器的 texel 获取量翻倍，使着色器更加昂贵。

这种 texel 获取的激增发生是因为您必须检索姿势矩阵的两个副本 - 每个动画一个 - 然后在它们之间进行混合。执行此操作的着色器代码可能如下代码片段所示：

```cpp
    mat4 pose0a = GetPose(animTexA, joints.x, instance);
    mat4 pose1a = GetPose(animTexA, joints.y, instance);
    mat4 pose2a = GetPose(animTexA, joints.z, instance);
    mat4 pose3a = GetPose(animTexA, joints.w, instance);
    mat4 pose0b = GetPose(animTexB, joints.x, instance);
    mat4 pose1b = GetPose(animTexB, joints.y, instance);
    mat4 pose2b = GetPose(animTexB, joints.z, instance);
    mat4 pose3b = GetPose(animTexB, joints.w, instance);
    mat4 pose0 = pose0a * (1.0 - fade) + pose0b * fade;
    mat4 pose1 = pose1a * (1.0 - fade) + pose1b * fade;
    mat4 pose2 = pose2a * (1.0 - fade) + pose2b * fade;
    mat4 pose3 = pose3a * (1.0 - fade) + pose3b * fade;
```

另一个原因是混合在技术上不正确。着色器在世界空间中进行线性混合。结果混合的骨架看起来不错，但与在本地空间中进行插值的关节不同。

如果你在两个姿势之间进行淡入淡出，混合是短暂的，只是为了隐藏过渡。在大多数情况下，过渡是否在技术上正确并不像过渡看起来平滑那样重要。在下一节中，您将探索使用替代纹理格式。

# 探索纹理格式

动画纹理目前以 32 位浮点纹理格式存储。这是一种容易存储动画纹理的格式，因为它与源数据的格式相同。这种方法在移动硬件上效果不佳。从主内存到图块内存的内存带宽是一种稀缺资源。

为了针对移动平台，考虑从`GL_RGBA32F`更改为带有`GL_UNSIGNED_BYTE`存储类型的`GL_RGBA`。切换到标准纹理格式确实意味着丢失一些数据。使用`GL_UNSIGNED_BYTE`存储类型，颜色的每个分量都限制在 256 个唯一值。这些值在采样时被标准化，并将返回在 0 到 1 的范围内。

如果任何动画信息存储值不在 0 到 1 的范围内，数据将需要被标准化。标准化比例因子将需要作为统一传递给着色器。如果你的目标是移动硬件，你可能只想存储旋转信息，这些信息已经在 0 到 1 的范围内。

在下一节中，您将探索如何将多个动画纹理合并成单个纹理。这减少了需要绑定的纹理数量，以便人群播放多个动画。

# 合并动画纹理

将许多较小的纹理合并成一个较大的纹理的行为称为纹理合并。包含多个较小纹理的大纹理通常称为纹理图集。纹理合并的好处是需要使用较少的纹理采样器。

本章介绍的人群渲染系统有一个主要缺点：虽然人群可以以不同的时间偏移播放动画，但他们只能播放相同的动画。有一个简单的方法可以解决这个问题：将多个动画纹理合并到一个大纹理上。

例如，一个*1024x1024*的纹理可以包含 16 个较小的*256x256*纹理。这意味着人群中的任何成员都可以播放 16 种动画中的一种。着色器的每个实例数据都需要添加一个额外的“偏移”统一。这个偏移统一将是一个`MAX_INSTANCES`大小的数组。

对于每个被渲染的角色，`GetPose`函数在检索动画纹素之前必须应用偏移。在下一节中，您将探索不同的技术，可以使用这些技术来通过最小化纹素获取来优化人群着色器。

# 优化纹素获取

即使在游戏 PC 上，渲染超过 200 个人群角色将花费超过 4 毫秒的时间，这是一个相当长的时间，假设您有 16.6 毫秒的帧时间。那么，为什么人群渲染如此昂贵呢？

每次调用`GetPose`辅助函数时，着色器执行 6 个纹素获取。由于每个顶点都被蒙皮到四个影响，每个顶点需要 24 个纹素获取！即使是低多边形模型，这也是大量的纹素获取。优化这个着色器将归结为最小化纹素获取的数量。

以下部分介绍了您可以使用的不同策略，以最小化每个顶点的纹素获取数量。

## 限制影响

优化纹素获取的一种天真的方法是在着色器代码中添加一个分支。毕竟，如果矩阵的权重为 0，为什么要获取姿势呢？这种优化可以实现如下：

```cpp
    mat4 pose0 = (weights.x < 0.0001)? 
        mat4(1.0) : GetPose(joints.x, instance);
    mat4 pose1 = (weights.y < 0.0001)? 
        mat4(1.0) : GetPose(joints.y, instance);
    mat4 pose2 = (weights.z < 0.0001)? 
        mat4(1.0) : GetPose(joints.z, instance);
    mat4 pose3 = (weights.w < 0.0001)? 
        mat4(1.0) : GetPose(joints.w, instance);
```

在最理想的情况下，这可能会节省一点时间。在最坏的情况下（每个骨骼恰好有四个影响），这实际上会给着色器增加额外的成本，因为现在每个影响都带有一个条件分支。

限制纹理获取的更好方法是限制骨骼影响。Blender、3DS Max 或 Maya 等 3DCC 工具具有导出选项，可以限制每个顶点的最大骨骼影响数量。您应该将最大骨骼影响数量限制为 1 或 2。

通常，在人群中，很难看清个别演员的细节。因此，将骨骼影响降低到 1，有效地使人群的皮肤刚性化，通常是可行的。在接下来的部分，您将探讨如何通过限制动画组件的数量来帮助减少每个顶点的纹理获取次数。

## 限制动画组件

考虑一个动画的人类角色。人类关节只旋转；它们永远不会平移或缩放。如果您知道一个动画只对每个关节进行一到两个组件的动画，`GetPose`函数可以被编辑以采样更少的数据。

这里还有一个额外的好处：可以将编码到动画纹理中的骨骼数量增加。如果您编码位置、旋转和缩放，最大关节数为`纹理大小/3`。如果您只编码一个组件，可以编码的关节数就是纹理的大小。

这种优化将使*256x256*纹理能够编码 256 个旋转，而不是 85 个变换。在下一节中，您将探讨是否需要在帧之间进行插值。

## 不进行插值

考虑动画纹理。它以设定的增量对动画进行采样，以填充纹理的每一列。在 256 个样本中，您可以在 60 FPS 下编码 3.6 秒的动画。

是否需要插值将取决于动画纹理的大小和被编码的动画长度。对于大多数游戏角色动画，如奔跑、行走、攻击或死亡，不需要帧插值。

通过这种优化，发送到 GPU 的数据量大大减少。帧统一可以从`ivec2`变为`int`，将数据大小减半。这意味着时间统一可以完全消失。

在下一节中，您将探讨您刚刚学到的三种优化的综合效果。

## 结合这些优化

让我们探讨这些优化可能产生的影响，假设以下三种优化都已实施：

+   将骨骼影响的数量限制为 2。

+   只对变换的旋转组件进行动画。

+   不要在帧之间进行插值。

这将把每个顶点的纹理获取次数从 24 减少到 2。可以编码到动画纹理中的关节数量将增加，并且每帧传输到 GPU 的数据量将大大减少。

# 总结

在本章中，您学会了如何将动画数据编码到纹理中，以及如何在顶点着色器中解释数据。还介绍了通过改变动画数据编码方式来改善性能的几种策略。将数据写入纹理的这种技术可以用于烘焙任何类型的采样数据。

要烘焙动画，您需要将其剪辑到纹理中。这个剪辑是在设定的间隔内采样的。每个骨骼的全局位置在每个间隔都被记录并写入纹理。在这个动画纹理中，每个关节占据三行：一个用于位置，一个用于旋转，一个用于缩放。

您使用实例化渲染了人群网格，并创建了一个可以从统一数组中读取每个实例数据的着色器。人群演员的每个实例数据，如位置、旋转和缩放，都作为统一数组传递给着色器，并使用实例 ID 作为这些数组的索引进行解释。

最后，您创建了`Crowd`类。这个实用类提供了一个易于使用的界面，用于管理人群中的演员。这个类将自动填充人群着色器的每个实例统一。使用这个类，您可以轻松地创建大型、有趣的人群。

本书的可下载内容中有本章的两个示例。`Sample00`是本章中我们编写的所有代码。另一方面，`Sample01`演示了如何在实践中使用这些代码来渲染大规模人群。
