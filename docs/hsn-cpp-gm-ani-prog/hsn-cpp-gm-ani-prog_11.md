# 第十一章：*第十一章*：优化动画管线

到目前为止，您已经编写了一个完整的动画系统，可以加载标准文件格式 gLT，并在 CPU 或 GPU 上执行皮肤。动画系统对于大多数简单的动画表现得足够好。

在本章中，您将探讨优化动画系统的方法，使其更快且资源消耗更少。这涉及探索执行皮肤的替代方法，提高采样动画片段的速度，并重新审视如何生成矩阵调色板。

每个主题都是单独探讨的，您可以选择实现尽可能少或尽可能多的这些优化。所有这些都很简单，可以轻松地用来替换不太优化的管线版本。

本章将涵盖以下主题：

+   预生成皮肤矩阵

+   将皮肤调色板存储在纹理中

+   更快的采样

+   姿势调色板生成

+   探索`Pose::GetGlobalTransform`

# 预生成皮肤矩阵

`mat4`对象的一个较大问题是占用了四个统一槽位，而经过处理的顶点着色器目前有两个具有 120 个元素的矩阵数组。总共是 960 个统一槽位，这是过多的。

顶点着色器中的这两个矩阵数组会发生什么？它们会相互相乘，如下所示：

```cpp
mat4 skin=(pose[joints.x]*invBindPose[joints.x])*weights.x;
  skin += (pose[joints.y]*invBindPose[joints.y])*weights.y;
  skin += (pose[joints.z]*invBindPose[joints.z])*weights.z;
  skin += (pose[joints.w]*invBindPose[joints.w])*weights.w;
```

这里的一个简单优化是将`pose * invBindPose`相乘，以便着色器只需要一个数组。这确实意味着一些皮肤过程被移回到了 CPU，但这个改变清理了 480 个统一槽位。

## 生成皮肤矩阵

生成皮肤矩阵不需要 API 调用-它很简单。使用`Pose`类的`GetMatrixPalette`函数从当前动画姿势生成矩阵调色板。然后，将调色板中的每个矩阵与相同索引的逆绑定姿势矩阵相乘。

显示网格的代码负责计算这些矩阵。例如，一个简单的更新循环可能如下所示：

```cpp
void Sample::Update(float deltaTime) {
    mPlaybackTime = mAnimClip.Sample(mAnimatedPose, 
                         mPlaybackTime + deltaTime);
    mAnimatedPose.GetMatrixPalette(mPosePalette);
    vector<mat4>& invBindPose = mSkeleton.GetInvBindPose();
    for (int i = 0; i < mPosePalette.size(); ++i) {
        mPosePalette[i] = mPosePalette[i] * invBindPose[i];
    }
    if (mDoCPUSkinning) {
        mMesh.CPUSkin(mPosePalette);
    }
}
```

在前面的代码示例中，动画片段被采样到一个姿势中。姿势被转换为矩阵向量。该向量中的每个矩阵然后与相同索引的逆绑定姿势矩阵相乘。结果的矩阵向量就是组合的皮肤矩阵。

如果网格是 CPU 皮肤，这是调用`CPUSkin`函数的好地方。这个函数需要重新实现以适应组合的皮肤矩阵。如果网格是 GPU 皮肤，需要编辑着色器以便只使用一个矩阵数组，并且需要更新渲染代码以便只传递一个统一数组。

在接下来的部分，您将探讨如何重新实现`CPUSkin`函数，使其与组合的皮肤矩阵一起工作。这将稍微加快 CPU 皮肤过程。

## CPU 皮肤

您需要一种新的皮肤方法，该方法尊重预乘的皮肤矩阵。此函数接受一个矩阵向量的引用。每个位置都由影响它的四个骨骼的组合皮肤矩阵进行变换。然后，这四个结果被缩放并相加。

将以下 CPU 皮肤函数添加到`Mesh.cpp`。不要忘记将函数声明添加到`Mesh.h`中：

1.  通过确保网格有效来开始实现`CPUSkin`函数。有效的网格至少有一个顶点。确保`mSkinnedPosition`和`mSkinnedNormal`向量足够大，可以容纳所有顶点：

```cpp
void Mesh::CPUSkin(std::vector<mat4>& animatedPose) {
    unsigned int numVerts = mPosition.size();
    if (numVerts == 0) { 
        return; 
    }
    mSkinnedPosition.resize(numVerts);
    mSkinnedNormal.resize(numVerts);
```

1.  接下来，循环遍历网格中的每个顶点：

```cpp
    for (unsigned int i = 0; i < numVerts; ++i) {
        ivec4& j = mInfluences[i];
        vec4& w = mWeights[i];
```

1.  将每个顶点按动画姿势变换四次，即每个影响顶点的关节变换一次。要找到经过处理的顶点，请将每个变换后的顶点按适当的权重进行缩放并将结果相加：

```cpp
        vec3 p0 = transformPoint(animatedPose[j.x], 
                                 mPosition[i]);
        vec3 p1 = transformPoint(animatedPose[j.y], 
                                 mPosition[i]);
        vec3 p2 = transformPoint(animatedPose[j.z], 
                                 mPosition[i]);
        vec3 p3 = transformPoint(animatedPose[j.w],
                                 mPosition[i]);
        mSkinnedPosition[i] = p0 * w.x + p1 * w.y + 
                              p2 * w.z + p3 * w.w;
```

1.  以相同的方式找到顶点的经过处理的法线：

```cpp
        vec3 n0 = transformVector(animatedPose[j.x], 
                                  mNormal[i]);
        vec3 n1 = transformVector(animatedPose[j.y], 
                                  mNormal[i]);
        vec3 n2 = transformVector(animatedPose[j.z], 
                                  mNormal[i]);
        vec3 n3 = transformVector(animatedPose[j.w], 
                                  mNormal[i]);
        mSkinnedNormal[i] = n0 * w.x + n1 * w.y + 
                            n2 * w.z + n3 * w.w;
    }
```

1.  通过将经过处理的顶点位置和经过处理的顶点法线上传到位置和法线属性来完成函数：

```cpp
    mPosAttrib->Set(mSkinnedPosition);
    mNormAttrib->Set(mSkinnedNormal);
}
```

核心的皮肤算法保持不变；唯一改变的是如何生成变换后的位置。现在，这个函数可以直接使用已经组合好的矩阵，而不必再组合动画姿势和逆绑定姿势。

在下一节中，您将探索如何将这个皮肤函数移入顶点着色器。动画和逆绑定姿势的组合仍然在 CPU 上完成，但实际顶点的皮肤可以在顶点着色器中实现。

## GPU 皮肤

在顶点着色器中实现预乘皮肤矩阵皮肤很简单。用新的预乘皮肤姿势替换姿势和逆绑定姿势的输入统一变量。使用这个新的统一数组生成皮肤矩阵。就是这样——其余的皮肤流程保持不变。

创建一个新文件`preskinned.vert`，来实现新的预皮肤顶点着色器。将`skinned.vert`的内容复制到这个新文件中。按照以下步骤修改新的着色器：

1.  旧的皮肤顶点着色器具有姿势和逆绑定姿势的统一变量。这两个统一变量都是矩阵数组。删除这些统一变量：

```cpp
uniform mat4 pose[120];
uniform mat4 invBindPose[120];
```

1.  用新的`animated`统一替换它们。这是一个矩阵数组，数组中的每个元素都包含`animated`姿势和逆绑定姿势矩阵相乘的结果。

```cpp
uniform mat4 animated[120];
```

1.  接下来，找到生成皮肤矩阵的位置。生成皮肤矩阵的代码如下：

```cpp
mat4 skin = (pose[joints.x] * invBindPose[joints.x]) *
             weights.x;
    skin += (pose[joints.y] * invBindPose[joints.y]) * 
             weights.y;
    skin += (pose[joints.z] * invBindPose[joints.z]) * 
             weights.z;
    skin += (pose[joints.w] * invBindPose[joints.w]) * 
             weights.w;
```

1.  用新的`animated`统一替换这个。对于影响顶点的每个关节，按适当的权重缩放`animated`统一矩阵并求和结果：

```cpp
mat4 skin = animated[joints.x] * weights.x +
            animated[joints.y] * weights.y +
            animated[joints.z] * weights.z +
            animated[joints.w] * weights.w;
```

着色器的其余部分保持不变。您需要更新的唯一内容是着色器接受的统一变量以及如何生成`skin`矩阵。在渲染时，`animated`矩阵可以设置如下：

```cpp
// mPosePalette Generated in the Update method!
int animated = mSkinnedShader->GetUniform("animated")
Uniform<mat4>::Set(animated, mPosePalette);
```

您可能已经注意到 CPU 皮肤实现和 GPU 皮肤实现是不同的。CPU 实现将顶点转换四次，然后缩放和求和结果。GPU 实现缩放和求和矩阵，只转换顶点一次。这两种实现都是有效的，它们都产生相同的结果。

在接下来的部分中，您将探索如何避免使用统一矩阵数组进行皮肤。

# 在纹理中存储皮肤调色板

预生成的皮肤矩阵可以减少所需的统一槽数量，但可以将所需的统一槽数量减少到一个。这可以通过在纹理中编码预生成的皮肤矩阵并在顶点着色器中读取该纹理来实现。

到目前为止，在本书中，您只处理了`RGB24`和`RGBA32`纹理。在这些格式中，每个像素的三个或四个分量使用每个分量 8 位编码。这只能容纳 256 个唯一值。这些纹理无法提供存储浮点数所需的精度。

这里还有另一种可能有用的纹理格式——`FLOAT32`纹理。使用这种纹理格式，向量的每个分量都得到一个完整的 32 位浮点数支持，给您完整的精度。这种纹理可以通过一个特殊的采样器函数进行采样，该函数不对数据进行归一化。`FLOAT32`纹理可以被视为 CPU 可以写入而 GPU 可以读取的缓冲区。

这种方法的好处是所需的统一槽数量变成了一个——所需的统一槽是`FLOAT32`纹理的采样器。缺点是速度。对每个顶点进行纹理采样比快速统一数组查找更昂贵。请记住，每次采样查找都需要返回几个 32 位浮点数。这是大量的数据要传输。

我们不会在这里涵盖存储皮肤矩阵的纹理的实现，因为在*第十五章*“使用实例渲染大规模人群”中有一个专门讨论这个主题的大节，其中包括完整的代码实现。

# 更快的采样

当前的动画剪辑采样代码表现良好，只要每个动画持续时间不超过 1 秒。但是，对于多个长达一分钟的动画剪辑，比如过场动画，动画系统的性能开始受到影响。为什么随着动画时间的增长性能会变差呢？罪魁祸首是`Track::FrameIndex`函数中的以下代码：

```cpp
    for (int i = (int)size - 1; i >= 0; --i) {
        if (time >= mFrames[i].mTime) {
            return i;
        }
    }
```

所呈现的循环遍历了轨道中的每一帧。如果动画有很多帧，性能就会变差。请记住，这段代码是针对动画剪辑中每个动画骨骼的每个动画组件执行的。

这个函数目前进行的是线性搜索，但可以通过更有效的搜索进行优化。由于时间只会增加，执行二分搜索是一个自然的优化。然而，二分搜索并不是最好的优化方法。可以将这个循环转换为常量查找。

采样动画的播放成本是统一的，不受长度的影响。它们在已知的采样间隔时间内计时每一帧，并且找到正确的帧索引只是将提供的时间归一化并将其移动到采样间隔范围内。不幸的是，这样的动画采样占用了大量内存。

如果你仍然按照给定的间隔对动画轨道进行采样，但是每个间隔不再包含完整的姿势，而是指向其左右的关键帧呢？采用这种方法，额外的内存开销是最小的，找到正确的帧是恒定的。

## 优化 Track 类

有两种方法可以优化`Track`类。你可以创建一个具有大部分`Track`类功能并维护已知采样时间的查找表的新类，或者扩展`Track`类。本节采用后一种方法——我们将扩展`Track`类。

`FastTrack`子类包含一个无符号整数向量。`Track`类以统一的时间间隔进行采样。对于每个时间间隔，播放头左侧的帧（即时间之前的帧）被记录到这个向量中。

所有新代码都添加到现有的`Track.h`和`Track.cpp`文件中。按照以下步骤实现`FastTrack`类：

1.  找到`Track`类的`FrameIndex`成员函数，并将其标记为`virtual`。这个改变允许新的子类重新实现`FrameIndex`函数。更新后的声明应该是这样的：

```cpp
template<typename T, int N>
class Track {
// ...
        virtual int FrameIndex(float time, bool looping);
// ...
```

1.  创建一个新类`FastTrack`，它继承自`Track`。`FastTrack`类包含一个无符号整数向量，重载的`FrameIndex`函数和一个用于填充无符号整数向量的函数：

```cpp
template<typename T, int N>
class FastTrack : public Track<T, N> {
protected:
    std::vector<unsigned int> mSampledFrames;
    virtual int FrameIndex(float time, bool looping);
public:
    void UpdateIndexLookupTable();
};
```

1.  为了使`FastTrack`类更易于使用，使用 typedef 为标量、向量和四元数类型创建别名：

```cpp
typedef FastTrack<float, 1> FastScalarTrack;
typedef FastTrack<vec3, 3> FastVectorTrack;
typedef FastTrack<quat, 4> FastQuaternionTrack;
```

1.  在`.cpp`文件中，为标量、向量和四元数的快速轨道添加模板声明：

```cpp
template FastTrack<float, 1>;
template FastTrack<vec3, 3>;
template FastTrack<quat, 4>;
```

由于`FastTrack`类是`Track`的子类，现有的 API 都可以不变地工作。通过以这种方式实现轨道采样，当涉及的动画帧数更多时，性能提升更大。在下一节中，你将学习如何构建索引查找表。

### 实现 UpdateIndexLookupTable

`UpdateIndexLookupTable`函数负责填充`mSampledFrames`向量。这个函数需要以固定的时间间隔对动画进行采样，并记录每个间隔的动画时间之前的帧。

`FastTrack`类应包含多少个样本？这个问题非常依赖于上下文，因为不同的游戏有不同的要求。对于本书的上下文来说，每秒 60 个样本应该足够了：

1.  通过确保轨道有效来开始实现`UpdateIndexLookupTable`函数。有效的轨道至少有两帧：

```cpp
template<typename T, int N>
void FastTrack<T, N>::UpdateIndexLookupTable() {
    int numFrames = (int)this->mFrames.size();
    if (numFrames <= 1) {
        return;
    }
```

1.  接下来，找到所需的样本数。由于每秒动画类有`60`个样本，将持续时间乘以`60`：

```cpp
    float duration = this->GetEndTime() - 
                     this->GetStartTime();
    unsigned int numSamples = duration * 60.0f;
    mSampledFrames.resize(numSamples);
```

1.  对于每个样本，找到沿着轨道的样本时间。要找到时间，将标准化迭代器乘以动画持续时间，并将动画的起始时间加上去：

```cpp
    for (unsigned int i = 0; i < numSamples; ++i) {
        float t = (float)i / (float)(numSamples - 1);
        float time = t*duration+this->GetStartTime();
```

1.  最后，是时候为每个给定的时间找到帧索引了。找到在此迭代中采样时间之前的帧，并将其记录在`mSampledFrames`向量中。如果采样帧是最后一帧，则返回最后一个索引之前的索引。请记住，`FrameIndex`函数永远不应返回最后一帧：

```cpp
        unsigned int frameIndex = 0;
        for (int j = numFrames - 1; j >= 0; --j) {
            if (time >= this->mFrames[j].mTime) {
                frameIndex = (unsigned int)j;
                if ((int)frameIndex >= numFrames - 2) {
                    frameIndex = numFrames - 2;
                }
                break;
            }
        }
        mSampledFrames[i] = frameIndex;
    }
}
```

`UpdateIndexLookupTable`函数旨在在加载时调用。通过记住内部`j`循环的上次使用的索引，可以优化它，因为在每次`i`迭代时，帧索引只会增加。在下一节中，您将学习如何实现`FrameIndex`以使用`mSampledFrames`向量。

### 实现 FrameIndex

`FrameIndex`函数负责找到给定时间之前的帧。优化的`FastTrack`类使用查找数组而不是循环遍历轨道的每一帧。所有输入时间的性能成本非常相似。按照以下步骤重写`FastTrack`类中的`FrameIndex`函数：

1.  通过确保轨道有效来开始实现`FrameIndex`函数。有效的轨道必须至少有两帧或更多：

```cpp
template<typename T, int N>
int FastTrack<T,N>::FrameIndex(float time,bool loop){
    std::vector<Frame<N>>& frames = this->mFrames;
    unsigned int size = (unsigned int)frames.size();
    if (size <= 1) { 
        return -1; 
}
```

1.  接下来，确保请求的采样时间落在轨道的起始时间和结束时间之间。如果轨道循环，使用`fmodf`来保持在有效范围内：

```cpp
    if (loop) {
        float startTime = this->mFrames[0].mTime;
        float endTime = this->mFrames[size - 1].mTime;
        float duration = endTime - startTime;
        time = fmodf(time - startTime, 
                     endTime - startTime);
        if (time < 0.0f) {
            time += endTime - startTime;
        }
        time = time + startTime;
    }
```

1.  如果轨道不循环，将其夹紧到第一帧或倒数第二帧：

```cpp
    else {
        if (time <= frames[0].mTime) {
            return 0;
        }
        if (time >= frames[size - 2].mTime) {
            return (int)size - 2;
        }
    }
```

1.  找到标准化的采样时间和帧索引。帧索引是标准化的采样时间乘以样本数。如果索引无效，则返回`-1`；否则返回索引指向的帧：

```cpp
    float duration = this->GetEndTime() - 
                     this->GetStartTime();
    float t = time / duration;
    unsigned int numSamples = (duration * 60.0f);
    unsigned int index = (t * (float)numSamples);
    if (index >= mSampledFrames.size()) {
        return -1;
    }
    return (int)mSampledFrames[index];
}
```

`FrameIndex`函数几乎总是在有效时间调用，因为它是一个受保护的辅助函数。这意味着找到帧索引所需的时间是均匀的，不管轨道中有多少帧。在下一节中，您将学习如何将未优化的`Track`类转换为优化的`FastTrack`类。

## 转换轨道

现在`FastTrack`存在了，如何创建它呢？您可以创建一个新的加载函数，加载`FastTrack`类而不是`Track`。或者，您可以创建一个将现有的`Track`类转换为`FastTrack`类的函数。本章采用后一种方法。按照以下步骤创建一个将`Track`对象转换为`FastTrack`对象的函数：

1.  在`FastTrack.h`中声明`OptimizeTrack`函数。该函数是模板化的。它接受与`Track`相同的模板类型：

```cpp
template<typename T, int N>
FastTrack<T, N> OptimizeTrack(Track<T, N>& input);
```

1.  在`FastTrack.cpp`中声明`OptimizeTrack`函数的模板特化，以适用于跟踪到`FastTrack`的所有三种类型。这意味着声明适用于标量、三维向量和四元数轨道的特化：

```cpp
template FastTrack<float, 1> 
OptimizeTrack(Track<float, 1>& input);
template FastTrack<vec3, 3> 
OptimizeTrack(Track<vec3, 3>& input);
template FastTrack<quat, 4> 
OptimizeTrack(Track<quat, 4>& input);
```

1.  要实现`OptimizeTrack`函数，调整结果轨道的大小，使其与输入轨道的大小相同并匹配插值。可以使用重载的`[]`运算符函数来复制每帧的数据：

```cpp
template<typename T, int N>
FastTrack<T, N> OptimizeTrack(Track<T, N>& input) {
    FastTrack<T, N> result;
    result.SetInterpolation(input.GetInterpolation());
    unsigned int size = input.Size();
    result.Resize(size);
    for (unsigned int i = 0; i < size; ++i) {
        result[i] = input[i];
    }
    result.UpdateIndexLookupTable();
    return result;
}
```

仅仅将`Track`类优化为`FastTrack`还不够。`TransformTrack`类也需要改变。它需要包含新的、优化的`FastTrack`类。在下一节中，您将更改`TransformTrack`类，使其成为模板，并且可以包含`Track`或`FastTrack`。

## 创建 FastTransformTrack

使用`Track`类的高级结构，如`TransformTrack`，需要适应新的`FastTrack`子类。`FastTrack`类与`Track`类具有相同的签名。因为类的签名相同，很容易将`TransformTrack`类模板化，以便它可以使用这两个类中的任何一个。

在这一部分，您将把`TransformTrack`类的名称更改为`TTransformTrack`并对类进行模板化。然后，您将将模板特化 typedef 为`TransformTrack`和`FastTransformTrack`。这样，`TransformTrack`类保持不变，优化的变换轨迹使用相同的代码：

1.  将`TransformTrack`类的名称更改为`TTransformTrack`并对类进行模板化。模板接受两个参数——要使用的矢量轨迹的类型和四元数轨迹的类型。更新`mPosition`、`mRotation`和`mScale`轨迹以使用新的模板类型：

```cpp
template <typename VTRACK, typename QTRACK>
class TTransformTrack {
protected:
   unsigned int mId;
   VTRACK mPosition;
   QTRACK mRotation;
   VTRACK mScale;
public:
   TTransformTrack();
   unsigned int GetId();
   void SetId(unsigned int id);
   VTRACK& GetPositionTrack();
   QTRACK& GetRotationTrack();
   VTRACK& GetScaleTrack();
   float GetStartTime();
   float GetEndTime();
   bool IsValid();
   Transform Sample(const Transform& r,float t,bool l);
};
```

1.  将这个类 typedef 为`TransformTrack`，使用`VectorTrack`和`QuaternionTrack`作为参数。再次将其 typedef 为`FastTransformTrack`，使用`FastVectorTrack`和`FastQuaternionTrack`作为模板参数：

```cpp
typedef TTransformTrack<VectorTrack, 
    QuaternionTrack> TransformTrack;
typedef TTransformTrack<FastVectorTrack, 
    FastQuaternionTrack> FastTransformTrack;
```

1.  声明将`TransformTrack`转换为`FastTransformTrack`的优化函数：

```cpp
FastTransformTrack OptimizeTransformTrack(
                   TransformTrack& input);
```

1.  在`TransformTrack.cpp`中为`typedef`函数添加模板规范：

```cpp
template TTransformTrack<VectorTrack, QuaternionTrack>;
template TTransformTrack<FastVectorTrack, 
                         FastQuaternionTrack>;
```

1.  实现`OptimizeTransformTrack`函数。复制轨迹 ID，然后通过值复制各个轨迹：

```cpp
FastTransformTrack OptimizeTransformTrack(
                   TransformTrack& input) {
    FastTransformTrack result;
    result.SetId(input.GetId());
    result.GetPositionTrack()= OptimizeTrack<vec3, 3> (
                             input.GetPositionTrack());
    result.GetRotationTrack() = OptimizeTrack<quat, 4>(
                             input.GetRotationTrack());
    result.GetScaleTrack()  =  OptimizeTrack<vec3, 3> (
                                input.GetScaleTrack());
    return result;
}
```

因为`OptimizeTransformTrack`通过值复制实际轨迹数据，所以它可能会有点慢。这个函数打算在初始化时调用。在下一节中，您将对`Clip`类进行模板化，类似于您对`Transform`类的操作，以创建`FastClip`。

## 创建 FastClip

这个动画系统的用户与`Clip`对象进行交互。为了适应新的`FastTrack`类，`Clip`类同样被模板化并分成了`Clip`和`FastClip`。您将实现一个函数来将`Clip`对象转换为`FastClip`对象。按照以下步骤对`Clip`类进行模板化：

1.  将`Clip`类的名称更改为`TClip`并对类进行模板化。模板只接受一种类型——`TClip`类包含的变换轨迹的类型。更改`mTracks`的类型和`[] operator`的返回类型，使其成为模板类型：

```cpp
template <typename TRACK>
class TClip {
protected:
    std::vector<TRACK> mTracks;
    std::string mName;
    float mStartTime;
    float mEndTime;
    bool mLooping;
public:
    TClip();
    TRACK& operator[](unsigned int index);
// ...
```

1.  使用`TransformTrack`类型将`TClip`typedef 为`Clip`。使用`FastTransformTrack`类型将`TClip`typedef 为`FastClip`。这样，`Clip`类不会改变，而`FastClip`类可以重用所有现有的代码：

```cpp
typedef TClip<TransformTrack> Clip;
typedef TClip<FastTransformTrack> FastClip;
```

1.  声明一个将`Clip`对象转换为`FastClip`对象的函数：

```cpp
FastClip OptimizeClip(Clip& input);
```

1.  在`Clip.cpp`中声明这些 typedef 类的模板特化：

```cpp
template TClip<TransformTrack>;
template TClip<FastTransformTrack>;
```

1.  要实现`OptimizeClip`函数，复制输入剪辑的名称和循环值。对于剪辑中的每个关节，调用其轨迹上的`OptimizeTransformTrack`函数。在返回副本之前，不要忘记计算新的`FastClip`对象的持续时间：

```cpp
FastClip OptimizeClip(Clip& input) {
    FastClip result;
    result.SetName(input.GetName());
    result.SetLooping(input.GetLooping());
    unsigned int size = input.Size();
    for (unsigned int i = 0; i < size; ++i) {
        unsigned int joint = input.GetIdAtIndex(i);
        result[joint] = 
              OptimizeTransformTrack(input[joint]);
    }
    result.RecalculateDuration();
    return result;
}
```

与其他转换函数一样，`OptimizeClip`只打算在初始化时调用。在接下来的部分，您将探讨如何优化`Pose`调色板的生成。

# 姿势调色板生成

您应该考虑的最终优化是从`Pose`生成矩阵调色板的过程。如果您查看`Pose`类，下面的代码将一个姿势转换为矩阵的线性数组：

```cpp
void Pose::GetMatrixPalette(std::vector<mat4>& out) {
    unsigned int size = Size();
    if (out.size() != size) {
        out.resize(size);
    }
    for (unsigned int i = 0; i < size; ++i) {
        Transform t = GetGlobalTransform(i);
        out[i] = transformToMat4(t);
    }
}
```

单独看，这个函数并不太糟糕，但`GetGlobalTransform`函数会循环遍历每个关节，一直到根关节的指定关节变换链。这意味着该函数会浪费大量时间来查找在上一次迭代期间已经找到的变换矩阵。

要解决这个问题，您需要确保`Pose`类中关节的顺序是升序的。也就是说，所有父关节在`mJoints`数组中的索引必须低于它们的子关节。

一旦设置了这个顺序，你可以遍历所有的关节，并知道当前索引处的关节的父矩阵已经找到。这是因为所有的父元素的索引都比它们的子节点小。为了将该关节的局部矩阵与其父关节的全局矩阵合并，你只需要将之前找到的世界矩阵和局部矩阵相乘。

不能保证输入数据可以信任地按照特定顺序列出关节。为了解决这个问题，你需要编写一些代码来重新排列`Pose`类的关节。在下一节中，你将学习如何改进`GetMatrixPalette`函数，使其在可能的情况下使用优化的方法，并在不可能的情况下退回到未优化的方法。

## 改变 GetMatrixPalette 函数

在本节中，你将修改`GetMatrixPalette`函数，以便在当前关节的父索引小于关节时预缓存全局矩阵。如果这个假设被打破，函数需要退回到更慢的计算模式。

`GetMatrixPalette`函数中将有两个循环。第一个循环找到并存储变换的全局矩阵。如果关节的父节点索引小于关节，就使用优化的方法。如果关节的父节点不小，第一个循环中断，并给第二个循环一个运行的机会。

在第二个循环中，每个关节都会退回到调用缓慢的`GetWorldTransform`函数来找到它的世界变换。如果优化的循环执行到最后，这个第二个循环就不会执行：

```cpp
void Pose::GetMatrixPalette(std::vector<mat4>& out) {
    int size = (int)Size();
    if ((int)out.size() != size) { out.resize(size); }
    int i = 0;
    for (; i < size; ++i) {
        int parent = mParents[i];
        if (parent > i) { break; }
        mat4 global = transformToMat4(mJoints[i]);
        if (parent >= 0) {
            global = out[parent] * global;
        }
        out[i] = global;
    }
    for (; i < size; ++i) {
        Transform t = GetGlobalTransform(i);
        out[i] = transformToMat4(t);
    }
}
```

这个改变对`GetMatrixPalette`函数的开销非常小，但很快就能弥补。它使得矩阵调色板计算运行快速，如果可能的话，但即使不可能也会执行。在接下来的部分，你将学习如何重新排列加载模型的关节，以便`GetMatrixPalette`函数始终采用快速路径。

## 重新排序关节

并非所有的模型都会格式良好；因此，它们不都能够利用优化的`GetMatrixPalette`函数。在本节中，你将学习如何重新排列模型的骨骼，以便它可以利用优化的`GetMatrixPalette`函数。

创建一个新文件`RearrangeBones.h`。使用一个字典，其键值对是骨骼索引和重新映射的骨骼索引。`RearrangeSkeleton`函数生成这个字典，并重新排列骨骼的绑定、逆绑定和静止姿势。

一旦`RearrangeSkeleton`函数生成了`BoneMap`，你可以使用它来处理任何影响当前骨骼的网格或动画片段。按照以下步骤重新排序关节，以便骨骼始终可以利用优化的`GetMatrixPalette`路径：

1.  将以下函数声明添加到`RearrangeBones.h`文件中：

```cpp
typedef std::map<int, int> BoneMap;
BoneMap RearrangeSkeleton(Skeleton& skeleton);
void RearrangeMesh(Mesh& mesh, BoneMap& boneMap);
void RearrangeClip(Clip& clip, BoneMap& boneMap);
void RearrangeFastclip(FastClip& clip, BoneMap& boneMap);
```

1.  在一个新文件`ReearrangeBones.cpp`中开始实现`RearrangeSkeleton`函数。首先，创建对静止和绑定姿势的引用，然后确保你要重新排列的骨骼不是空的。如果是空的，就返回一个空的字典：

```cpp
BoneMap RearrangeSkeleton(Skeleton& skeleton) {
    Pose& restPose = skeleton.GetRestPose();
    Pose& bindPose = skeleton.GetBindPose();
    unsigned int size = restPose.Size();
    if (size == 0) { return BoneMap(); }
```

1.  接下来，创建一个二维整数数组（整数向量的向量）。外部向量的每个元素代表一个骨骼，该向量和绑定或静止姿势中的`mJoints`数组的索引是平行的。内部向量表示外部向量索引处的关节包含的所有子节点。循环遍历静止姿势中的每个关节：

```cpp
    std::vector<std::vector<int>> hierarchy(size);
    std::list<int> process;
    for (unsigned int i = 0; i < size; ++i) {
        int parent = restPose.GetParent(i);
```

1.  如果一个关节有父节点，将该关节的索引添加到父节点的子节点向量中。如果一个节点是根节点（没有父节点），直接将其添加到处理列表中。稍后将使用该列表来遍历地图深度：

```cpp
        if (parent >= 0) {
            hierarchy[parent].push_back((int)i);
        }
        else {
            process.push_back((int)i);
        }
    }
```

1.  要弄清楚如何重新排序骨骼，你需要保留两个映射——一个从旧配置映射到新配置，另一个从新配置映射回旧配置：

```cpp
    BoneMap mapForward;
    BoneMap mapBackward;
```

1.  对于每个元素，如果它包含子元素，则将子元素添加到处理列表中。这样，所有的关节都被处理，层次结构中较高的关节首先被处理：

```cpp
    int index = 0;
    while (process.size() > 0) {
        int current = *process.begin();
        process.pop_front();
        std::vector<int>& children = hierarchy[current];
        unsigned int numChildren = children.size();
        for (unsigned int i = 0; i < numChildren; ++i) {
            process.push_back(children[i]);
        }
```

1.  将正向映射的当前索引设置为正在处理的关节的索引。正向映射的当前索引是一个原子计数器。对于反向映射也是同样的操作，但是要交换键值对。不要忘记将空节点（`-1`）添加到两个映射中：

```cpp
        mapForward[index] = current;
        mapBackward[current] = index;
        index += 1;
    }
    mapForward[-1] = -1;
    mapBackward[-1] = -1;
```

1.  现在映射已经填充，您需要构建新的静止和绑定姿势，使其骨骼按正确的顺序排列。循环遍历原始静止和绑定姿势中的每个关节，并将它们的本地变换复制到新的姿势中。对于关节名称也是同样的操作：

```cpp
    Pose newRestPose(size);
    Pose newBindPose(size);
    std::vector<std::string> newNames(size);
    for (unsigned int i = 0; i < size; ++i) {
        int thisBone = mapForward[i];
        newRestPose.SetLocalTransform(i, 
                restPose.GetLocalTransform(thisBone));
        newBindPose.SetLocalTransform(i, 
                bindPose.GetLocalTransform(thisBone));
        newNames[i] = skeleton.GetJointName(thisBone);
```

1.  为每个关节找到新的父关节 ID 需要两个映射步骤。首先，将当前索引映射到原始骨架中的骨骼。这将返回原始骨架的父关节。将此父索引映射回新骨架。这就是为什么有两个字典，以便进行快速映射：

```cpp
        int parent = mapBackward[bindPose.GetParent(
                                         thisBone)];
        newRestPose.SetParent(i, parent);
        newBindPose.SetParent(i, parent);
    }
```

1.  一旦找到新的静止和绑定姿势，并且关节名称已经相应地重新排列，通过调用公共的`Set`方法将这些数据写回骨架。骨架的`Set`方法还会计算逆绑定姿势矩阵调色板：

```cpp
    skeleton.Set(newRestPose, newBindPose, newNames);
    return mapBackward;
} // End of RearrangeSkeleton function
```

`RearrangeSkeleton`函数重新排列骨架中的骨骼，以便骨架可以利用`GetMatrixPalette`的优化版本。重新排列骨架是不够的。由于关节索引移动，引用该骨架的任何剪辑或网格现在都是损坏的。在下一节中，您将实现辅助函数来重新排列剪辑中的关节。

## 重新排序剪辑

要重新排列动画剪辑，循环遍历剪辑中的所有轨道。对于每个轨道，找到关节 ID，然后使用`RearrangeSkeleton`函数返回的（反向）骨骼映射转换该关节 ID。将修改后的关节 ID 写回到轨道中：

```cpp
void RearrangeClip(Clip& clip, BoneMap& boneMap) {
    unsigned int size = clip.Size();
    for (unsigned int i = 0; i < size; ++i) {
        int joint = (int)clip.GetIdAtIndex(i);
        unsigned int newJoint = (unsigned int)boneMap[joint];
        clip.SetIdAtIndex(i, newJoint);
    }
}
```

如果您之前在本章中实现了`FastClip`优化，`RearrangeClip`函数应该仍然有效，因为它是`Clip`的子类。在下一节中，您将学习如何重新排列网格中的关节，这将是使用此优化所需的最后一步。

## 重新排序网格

要重新排列影响网格蒙皮的关节，循环遍历网格的每个顶点，并重新映射该顶点的影响属性中存储的四个关节索引。关节的权重不需要编辑，因为关节本身没有改变；只是其数组中的索引发生了变化。

以这种方式更改网格只会编辑网格的 CPU 副本。调用`UpdateOpenGLBuffers`将新属性上传到 GPU：

```cpp
void RearrangeMesh(Mesh& mesh, BoneMap& boneMap) {
    std::vector<ivec4>& influences = mesh.GetInfluences();
    unsigned int size = (unsigned int)influences.size();
    for (unsigned int i = 0; i < size; ++i) {
        influences[i].x = boneMap[influences[i].x];
        influences[i].y = boneMap[influences[i].y];
        influences[i].z = boneMap[influences[i].z];
        influences[i].w = boneMap[influences[i].w];
    }
    mesh.UpdateOpenGLBuffers();
}
```

实现了`RearrangeMesh`函数后，您可以加载一个骨架，然后调用`RearrangeSkeleton`函数并存储它返回的骨骼映射。使用这个骨骼映射，您还可以使用`RearrangeClip`和`RearrangeMesh`函数修复引用骨架的任何网格或动画剪辑。经过这种方式处理后，`GetMatrixPalette`始终采用优化路径。在下一节中，您将探索在层次结构中缓存变换。

# 探索 Pose::GetGlobalTransform

`Pose`类的`GetGlobalTransform`函数的一个特点是它总是计算世界变换。考虑这样一种情况，您请求一个节点的世界变换，然后立即请求其父节点的世界变换。原始请求计算并使用父节点的世界变换，但一旦下一个请求被发出，同样的变换就会再次计算。

解决这个问题的方法是向`Pose`类添加两个新数组。一个是世界空间变换的向量，另一个包含脏标志。每当设置关节的本地变换时，关节的脏标志需要设置为`true`。

当请求世界变换时，会检查变换及其所有父级的脏标志。如果该链中有脏变换，则重新计算世界变换。如果脏标志未设置，则返回缓存的世界变换。

本章不会实现这个优化。这个优化会给`Pose`类的每个实例增加大量的内存。除了逆向运动学的情况，`GetGlobalTransform`函数很少被使用。对于蒙皮，`GetMatrixPalette`函数用于检索世界空间矩阵，而该函数已经被优化过了。

# 总结

在本章中，你探索了如何针对几种情况优化动画系统。这些优化减少了顶点蒙皮着色器所需的统一变量数量，加快了具有许多关键帧的动画的采样速度，并更快地生成了姿势的矩阵调色板。

请记住，没有一种大小适合所有的解决方案。如果游戏中的所有动画都只有几个关键帧，那么通过查找表优化动画采样所增加的开销可能不值得额外的内存。然而，改变采样函数以使用二分查找可能是值得的。每种优化策略都存在类似的利弊；你必须选择适合你特定用例的方案。

在查看本章的示例代码时，`Chapter11/Sample00`包含了本章的全部代码。`Chapter11/Sample01`展示了如何使用预蒙皮网格，`Chapter11/Sample02`展示了如何使用`FastTrack`类进行更快的采样，`Chapter11/Sample03`展示了如何重新排列骨骼以加快调色板的生成。

在下一章中，你将探索如何混合动画以平滑地切换两个动画。本章还将探讨修改现有动画的混合技术。
