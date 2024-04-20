# *第五章*：实现变换

在本章中，您将实现一个包含位置、旋转和缩放数据的结构。这个结构就是一个变换。变换将一个空间映射到另一个空间。位置、旋转和缩放也可以存储在 4x4 矩阵中，那么为什么要使用显式的变换结构而不是矩阵？答案是插值。矩阵的插值效果不好，但变换结构可以。

在两个矩阵之间进行插值是困难的，因为旋转和缩放存储在矩阵的相同组件中。因此，在两个矩阵之间进行插值不会产生您期望的结果。变换通过分别存储位置、旋转和缩放组件来解决了这个问题。

在本章中，您将实现一个变换结构以及您需要执行的常见操作。在本章结束时，您应该能够做到以下事情：

+   理解什么是变换

+   理解如何组合变换

+   在变换和矩阵之间进行转换

+   理解如何将变换应用到点和向量

重要信息

在本章中，您将实现一个表示位置、旋转和缩放的变换结构。要了解更多关于变换，它们与矩阵的关系以及它们如何适应游戏层次结构，请访问[`gabormakesgames.com/transforms.html`](http://gabormakesgames.com/transforms.html)。

# 创建变换。

变换是简单的结构。一个变换包含一个位置、旋转和缩放。位置和缩放是向量，旋转是四元数。变换可以按层次结构组合，但这种父子关系不应该是实际变换结构的一部分。以下步骤将指导您创建一个变换结构：

1.  创建一个新文件，`Transform.h`。这个文件是必需的，用来声明变换结构。

1.  在这个新文件中声明`Transform`结构。从变换的属性—`position`、`rotation`和`scale`开始：

```cpp
    struct Transform {
        vec3 position;
        quat rotation;
        vec3 scale;
    ```

1.  创建一个构造函数，它接受一个位置、旋转和缩放。这个构造函数应该将这些值分配给`Transform`结构的适当成员：

```cpp
    Transform(const vec3& p, const quat& r, const vec3& s) :
        position(p), rotation(r), scale(s) {}
    ```

1.  空变换不应该有位置或旋转，缩放为 1。默认情况下，`scale`组件将被创建为`(0, 0, 0)`。为了解决这个问题，`Transform`结构的默认构造函数需要将`scale`初始化为正确的值：

```cpp
        Transform() :
            position(vec3(0, 0, 0)),
            rotation(quat(0, 0, 0, 1)),
            scale(vec3(1, 1, 1))
        {}
    }; // End of transform struct
    ```

`Transform`结构非常简单；它的所有成员都是公共的。一个变换有一个位置、旋转和缩放。默认构造函数将位置向量设置为*0*，将旋转四元数设置为单位，将缩放向量设置为*1*。默认构造函数创建的变换没有效果。

在下一节中，您将学习如何以与矩阵或四元数类似的方式组合变换。

# 组合变换

以骨架为例。在每个关节处，您可以放置一个变换来描述关节的运动。当您旋转肩膀时，连接到该肩膀的肘部也会移动。要将肩部变换应用于所有连接的关节，必须将每个关节上的变换与其父关节的变换相结合。

变换可以像矩阵和四元数一样组合，并且两个变换的效果可以组合成一个变换。为保持一致，组合变换应保持从右到左的组合顺序。与矩阵和四元数不同，这个`combine`函数不会被实现为一个乘法函数。

组合两个变换的缩放和旋转很简单—将它们相乘。组合位置有点困难。组合位置需要受到`rotation`和`scale`组件的影响。在找到组合位置时，记住变换的顺序：先缩放，然后旋转，最后平移。

创建一个新文件，`Transform.cpp`。实现`combine`函数，并不要忘记将函数声明添加到`Transform.h`中：

```cpp
Transform combine(const Transform& a, const Transform& b) {
    Transform out;
    out.scale = a.scale * b.scale;
    out.rotation = b.rotation * a.rotation;
    out.position = a.rotation * (a.scale * b.position);
    out.position = a.position + out.position;
    return out;
}
```

在后面的章节中，`combine`函数将用于将变换组织成层次结构。在下一节中，你将学习如何反转变换，这与反转矩阵和四元数类似。

# 反转变换

你已经知道变换将一个空间映射到另一个空间。可以反转该映射，并将变换映射回原始空间。与矩阵和四元数一样，变换也可以被反转。

在反转缩放时，请记住 0 不能被反转。缩放为 0 的情况需要特殊处理。

在`Transform.cpp`中实现`inverse`变换方法。不要忘记在`Transform.h`中声明该方法：

```cpp
Transform inverse(const Transform& t) {
    Transform inv;
    inv.rotation = inverse(t.rotation);
    inv.scale.x = fabs(t.scale.x) < VEC3_EPSILON ? 
                  0.0f : 1.0f / t.scale.x;
    inv.scale.y = fabs(t.scale.y) < VEC3_EPSILON ? 
                  0.0f : 1.0f / t.scale.y;
    inv.scale.z = fabs(t.scale.z) < VEC3_EPSILON ? 
                  0.0f : 1.0f / t.scale.z;
    vec3 invTrans = t.position * -1.0f;
    inv.position = inv.rotation * (inv.scale * invTrans);
    return inv;
}
```

反转变换可以消除一个变换对另一个变换的影响。考虑一个角色在关卡中移动。一旦关卡结束，你可能希望将角色移回原点，然后开始下一个关卡。你可以将角色的变换乘以它的逆变换。

在下一节中，你将学习如何将两个或多个变换混合在一起。

# 混合变换

你有代表两个特定时间点的关节的变换。为了使模型看起来动画化，你需要在这些帧的变换之间进行插值或混合。

可以在向量和四元数之间进行插值，这是变换的构建块。因此，也可以在变换之间进行插值。这个操作通常被称为混合。当将两个变换混合在一起时，线性插值输入变换的位置、旋转和缩放。

在`Transform.cpp`中实现`mix`函数。不要忘记在`Transform.h`中声明该函数：

```cpp
Transform mix(const Transform& a,const Transform& b,float t){
    quat bRot = b.rotation;
    if (dot(a.rotation, bRot) < 0.0f) {
        bRot = -bRot;
    }
    return Transform(
        lerp(a.position, b.position, t),
        nlerp(a.rotation, bRot, t),
        lerp(a.scale, b.scale, t));
}
```

能够将变换混合在一起对于创建动画之间的平滑过渡非常重要。在这里，你实现了变换之间的线性混合。在下一节中，你将学习如何将`transform`转换为`mat4`。

# 将变换转换为矩阵

着色器程序与矩阵配合得很好。它们没有本地表示变换结构。你可以将变换代码转换为 GLSL，但这不是最好的解决方案。相反，你可以在将变换提交为着色器统一之前将变换转换为矩阵。

由于变换编码了可以存储在矩阵中的数据，因此可以将变换转换为矩阵。要将变换转换为矩阵，需要考虑矩阵的向量。

首先，通过将全局基向量的方向乘以变换的旋转来找到基向量。接下来，通过变换的缩放来缩放基向量。这将产生填充上 3x3 子矩阵的最终基向量。位置直接进入矩阵的最后一列。

在`Transform.cpp`中实现`from Transform`方法。不要忘记将该方法声明到`Transform.h`中：

```cpp
mat4 transformToMat4(const Transform& t) {
    // First, extract the rotation basis of the transform
    vec3 x = t.rotation * vec3(1, 0, 0);
    vec3 y = t.rotation * vec3(0, 1, 0);
    vec3 z = t.rotation * vec3(0, 0, 1);
    // Next, scale the basis vectors
    x = x * t.scale.x;
    y = y * t.scale.y;
    z = z * t.scale.z;
    // Extract the position of the transform
    vec3 p = t.position;
    // Create matrix
    return mat4(
        x.x, x.y, x.z, 0, // X basis (& Scale)
        y.x, y.y, y.z, 0, // Y basis (& scale)
        z.x, z.y, z.z, 0, // Z basis (& scale)
        p.x, p.y, p.z, 1  // Position
    );
}
```

图形 API 使用矩阵而不是变换。在后面的章节中，变换将在发送到着色器之前转换为矩阵。在下一节中，你将学习如何做相反的操作，即将矩阵转换为变换。

# 将矩阵转换为变换

外部文件格式可能将变换数据存储为矩阵。例如，glTF 可以将节点的变换存储为位置、旋转和缩放，或者作为单个 4x4 矩阵。为了使变换代码健壮，你需要能够将矩阵转换为变换。

将矩阵转换为变换比将变换转换为矩阵更困难。提取矩阵的旋转很简单；你已经实现了将 4x4 矩阵转换为四元数的函数。提取位置也很简单；将矩阵的最后一列复制到一个向量中。提取比例尺更困难。

回想一下，变换的操作顺序是先缩放，然后旋转，最后平移。这意味着如果你有三个矩阵——*S*、*R*和*T*——分别代表缩放、旋转和平移，它们将组合成一个变换矩阵*M*，如下所示：

*M = SRT*

要找到比例尺，首先忽略矩阵的平移部分*M*（将平移向量归零）。这样你就得到*M = SR*。要去除矩阵的旋转部分，将*M*乘以*R*的逆。这样应该只剩下比例尺部分。嗯，并不完全是这样。结果会留下一个包含比例尺和一些倾斜信息的矩阵。

我们从这个比例尺-倾斜矩阵中提取比例尺的方法是简单地将主对角线作为比例尺-倾斜矩阵。虽然这在大多数情况下都有效，但并不完美。获得的比例尺应该被视为有损的比例尺，因为该值可能包含倾斜数据，这使得比例尺不准确。

重要提示

将矩阵分解为平移、旋转、缩放、倾斜和行列式的符号是可能的。然而，这种分解是昂贵的，不太适合实时应用。要了解更多，请查看 Ken Shoemake 和 Tom Duff 的*Matrix Animation and Polar Decomposition* [`research.cs.wisc.edu/graphics/Courses/838-s2002/Papers/polar-decomp.pdf`](https://research.cs.wisc.edu/graphics/Courses/838-s2002/Papers/polar-decomp.pdf)。

在`Transform.cpp`中实现`toTransform`函数。不要忘记将函数声明添加到`Transform.h`中：

```cpp
Transform mat4ToTransform(const mat4& m) {
    Transform out;
    out.position = vec3(m.v[12], m.v[13], m.v[14]);
    out.rotation = mat4ToQuat(m);
    mat4 rotScaleMat(
        m.v[0], m.v[1], m.v[2], 0,
        m.v[4], m.v[5], m.v[6], 0,
        m.v[8], m.v[9], m.v[10], 0,
        0, 0, 0, 1
    );
    mat4 invRotMat = quatToMat4(inverse(out.rotation));
    mat4 scaleSkewMat = rotScaleMat * invRotMat;
    out.scale = vec3(
        scaleSkewMat.v[0], 
        scaleSkewMat.v[5], 
        scaleSkewMat.v[10]
    );
    return out;
}
```

能够将矩阵转换为变换是很重要的，因为你并不总是能控制你处理的数据以什么格式呈现。例如，一个模型格式可能存储矩阵而不是变换。

到目前为止，你可能已经注意到变换和矩阵通常可以做相同的事情。在下一节中，你将学习如何使用变换来对点和向量进行变换，类似于使用矩阵的方式。

# 变换点和向量

`Transform`结构可用于在空间中移动点和向量。想象一个球上下弹跳。球的弹跳是由`Transform`结构派生的，但你如何知道每个球的顶点应该移动到哪里？你需要使用`Transform`结构（或矩阵）来正确显示球的所有顶点。

使用变换来修改点和向量就像组合两个变换。要变换一个点，首先应用缩放，然后旋转，最后是变换的平移。要变换一个向量，遵循相同的步骤，但不要添加位置：

1.  在`Transform.cpp`中实现`transformPoint`函数。不要忘记将函数声明添加到`Transform.h`中：

```cpp
    vec3 transformPoint(const Transform& a, const vec3& b) {
        vec3 out;
        out = a.rotation * (a.scale * b);
        out = a.position + out;
        return out;
    }
    ```

1.  在`Transform.cpp`中实现`transformVector`函数。不要忘记将函数声明添加到`Transform.h`中：

```cpp
    vec3 transformVector(const Transform& a, const vec3& b) {
        vec3 out;
        out = a.rotation * (a.scale * b);
        return out;
    }
    ```

`transformPoint`函数做的就是一个一个步骤地将矩阵和点相乘。首先应用`scale`，然后是`rotation`，最后是`translation`。当处理向量而不是点时，同样的顺序适用，只是忽略了平移。

# 总结

在本章中，你学会了将变换实现为一个包含位置、旋转和比例尺的离散结构。在许多方面，`Transform`类保存了你通常会存储在矩阵中的相同数据。

你学会了如何组合、反转和混合变换，以及如何使用变换来移动点和旋转向量。变换在未来将是至关重要的，因为它们用于动画游戏模型的骨骼或骨架。

你需要一个显式的`Transform`结构的原因是矩阵不太容易插值。对变换进行插值对于动画非常重要。这是你创建中间姿势以显示两个给定关键帧的方式。

在下一章中，你将学习如何在 OpenGL 之上编写一个轻量级的抽象层，以使未来章节中的渲染更容易。
