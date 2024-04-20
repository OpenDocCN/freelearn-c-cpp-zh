# 第十二章：*第十二章*：动画之间的混合

从一个动画过渡到另一个动画可能会很突兀。想象一下，如果一个角色正在进行一次拳击，玩家决定开始奔跑。如果动画直接从跳跃片段切换到奔跑片段，过渡将会很生硬和不自然。

动画混合可以通过生成两个动画的平均中间帧来修复这个问题。这种淡入通常很短——不到一秒钟。这种短混合产生的平滑动画过渡提供了更好的观感体验。

本章探讨了如何实现动画混合和附加动画混合，以及如何设置交叉淡入淡出控制器来管理混合队列。将涵盖以下主题：

+   姿势混合

+   交叉淡入淡出动画

+   附加混合

# 姿势混合

动画混合是在每个关节的本地空间中两个姿势之间的线性混合。可以将其视为`lerp`或`mix`函数，但应用于整个姿势。这种技术不是混合动画片段；而是混合这些片段被采样到的姿势。

在混合两个姿势时，不需要整个姿势都进行混合。假设有两个动画——奔跑循环和攻击。如果玩家按下攻击按钮，攻击姿势的上半部分在短时间内混合进来，保持在整个动画中的权重为`1`，然后在动画结束时淡出。

这是一个使用姿势混合来创建奔跑攻击动画的示例，而无需对攻击动画的腿部进行动画处理。攻击动画可以在行走动画的基础上混合。动画混合可用于平滑地过渡动画或将多个动画组合成一个新动画。

在接下来的部分，您将为`Pose`类声明一个`Blend`函数。这个`Blend`函数将在两个姿势之间进行线性插值，类似于向量`lerp`的工作方式。该函数需要两个姿势和一个插值值，通常表示为`t`，其范围为`0`到`1`。

## 声明混合函数

`Blend`函数接受两个姿势——混合值和根节点作为参数。当混合值为`0`时，`Blend`函数返回第一个姿势，当为`1`时，返回第二个姿势。对于介于`0`和`1`之间的任何值，姿势都会被混合。根节点决定了第二个动画的哪个节点（及其子节点）应该混合到第一个动画中。

为了适应指定从哪个骨骼节点开始混合，需要一种方法来检查一个节点是否在另一个节点的层次结构中。`IsInHierarchy`函数接受一个`Pose`类，一个作为根节点的节点和一个作为搜索节点的节点。如果搜索节点是根节点的后代，则函数返回`true`：

```cpp
bool IsInHierarchy(Pose& pose, unsigned int root, 
                   unsigned int search);
void Blend(Pose& output,Pose& a,Pose& b,float t,int root);
```

当混合两个姿势时，假设这些姿势是相似的。相似的姿势具有相同数量的关节，并且每个关节在姿势之间具有相同的父级索引。在接下来的部分，您将实现`Blend`函数。

## 实现混合功能

为了使混合有效，它必须在本地空间中进行，这对于在两个姿势之间进行混合非常方便。循环遍历输入姿势中的所有关节，并在正在混合的两个姿势中插值关节的本地变换。对于位置和比例，使用向量`lerp`函数，对于旋转，使用四元数`nlerp`函数。

为了支持动画根节点，检查当前变换是否是混合根的后代。如果是，进行混合。如果不是，则跳过混合，并保持第一个输入姿势的变换值。按照以下步骤实现层次结构检查和`Blend`函数：

1.  要检查一个关节是否是另一个关节的后代，沿着后代关节一直向上遍历层次结构，直到根节点。如果在这个层次结构中遇到的任何节点都是您要检查的节点，则返回`true`：

```cpp
bool IsInHierarchy(Pose& pose, unsigned int parent, 
                   unsigned int search) {
    if (search == parent) {
        return true;
    }
    int p = pose.GetParent(search);
    while (p >= 0) {
        if (p == (int)parent) {
            return true;
        }
        p = pose.GetParent(p);
    }
    return false;
}
```

1.  为了将两个姿势混合在一起，循环遍历每个姿势的关节。如果当前关节不在混合根的层次结构中，则不进行混合。否则，使用您在*第五章*中编写的`mix`函数来混合`Transform`对象。`mix`函数考虑四元数邻域：

```cpp
void Blend(Pose& output, Pose& a, Pose& b, 
           float t, int root) {
    unsigned int numJoints = output.Size();
    for (unsigned int i = 0; i < numJoints; ++i) {
        if (root >= 0) {
            if (!IsInHierarchy(output, root, i)) {
                continue;
            }
        }
        output.SetLocalTransform(i, mix(
              a.GetLocalTransform(i), 
              b.GetLocalTransform(i), t)
        );
    }
}
```

如果使用整个层次结构混合两个动画，则`Blend`的根参数将为负数。对于混合根的负关节，`Blend`函数会跳过`IsInHierarchy`检查。在接下来的部分，您将探索如何在两个动画之间进行淡入淡出以实现平滑过渡。

# 淡入淡出动画

混合动画的最常见用例是在两个动画之间进行淡入淡出。**淡入淡出**是从一个动画快速混合到另一个动画。淡入淡出的目标是隐藏两个动画之间的过渡。

一旦淡入淡出完成，活动动画需要被正在淡入的动画替换。如果您正在淡入多个动画，则它们都会被评估。最先结束的动画首先被移除。请求的动画被添加到列表中，已经淡出的动画被从列表中移除。

在接下来的部分，您将构建一个`CrossFadeController`类来处理淡入淡出逻辑。这个类提供了一个简单直观的 API，只需一个函数调用就可以简单地在动画之间进行淡入淡出。

## 创建辅助类

当将动画淡入到已经采样的姿势中时，您需要知道正在淡入的动画是什么，它的当前播放时间，淡入持续时间的长度以及淡入的当前时间。这些值用于执行实际的混合，并包含有关混合状态的数据。

创建一个新文件并命名为`CrossFadeTarget.h`，以实现`CrossFadeTarget`辅助类。这个辅助类包含了之前描述的变量。默认构造函数应将所有值设置为`0`。还提供了一个方便的构造函数，它接受剪辑指针、姿势引用和持续时间：

```cpp
struct CrossFadeTarget {
   Pose mPose;
   Clip* mClip;
   float mTime;
   float mDuration;
   float mElapsed;
   inline CrossFadeTarget() 
          : mClip(0), mTime(0.0f), 
            mDuration(0.0f), mElapsed(0.0f) { }
   inline CrossFadeTarget(Clip* target,Pose& pose,float dur) 
          : mClip(target), mTime(target->GetStartTime()), 
            mPose(pose), mDuration(dur), mElapsed(0.0f) { }
};
```

`CrossFadeTarget`辅助类的`mPose`、`mClip`和`mTime`变量在每一帧都用于采样正在淡入的动画。`mDuration`和`mElapsed`变量用于控制动画应该淡入多少。

在下一节中，您将实现一个控制动画播放和淡入淡出的类。

## 声明淡入淡出控制器

跟踪当前播放的剪辑并管理淡入淡出是新的`CrossFadeController`类的工作。创建一个新文件`CrossFadeController.h`，声明新的类。这个类需要包含一个骨架、一个姿势、当前播放时间和一个动画剪辑。它还需要一个控制动画混合的`CrossFadeTarget`对象的向量。

`CrossFadeController`和`CrossFadeTarget`类都包含指向动画剪辑的指针，但它们不拥有这些指针。因为这两个类都不拥有指针的内存，所以生成的构造函数、复制构造函数、赋值运算符和析构函数应该可以正常使用。

`CrossFadecontroller`类需要函数来设置当前骨架、检索当前姿势和检索当前剪辑。当前动画可以使用`Play`函数设置。可以使用`FadeTo`函数淡入新动画。由于`CrossFadeController`类管理动画播放，它需要一个`Update`函数来采样动画剪辑：

```cpp
class CrossFadeController {
protected:
    std::vector<CrossFadeTarget> mTargets;
    Clip* mClip;
    float mTime;
    Pose mPose;
    Skeleton mSkeleton;
    bool mWasSkeletonSet;
public:
    CrossFadeController();
    CrossFadeController(Skeleton& skeleton);
    void SetSkeleton(Skeleton& skeleton);
    void Play(Clip* target);
    void FadeTo(Clip* target, float fadeTime);
    void Update(float dt);
    Pose& GetCurrentPose();
    Clip* GetcurrentClip();
};
```

整个`mTargets`列表在每一帧都会被评估。每个动画都会被评估并混合到当前播放的动画中。

在接下来的部分，您将实现`CrossFadeController`类。

## 实现淡出控制器

创建一个新文件，`CrossFadeController.cpp`。在这个新文件中实现`CrossFadeController`。按照以下步骤实现`CrossFadeController`：

1.  在默认构造函数中，为当前剪辑和时间设置默认值`0`，并将骨骼标记为未设置。还有一个方便的构造函数，它接受一个骨骼引用。方便的构造函数应调用`SetSkeleton`函数：

```cpp
CrossFadeController::CrossFadeController() {
    mClip = 0;
    mTime = 0.0f;
    mWasSkeletonSet = false;
}
CrossFadeController::CrossFadeController(Skeleton& skeleton) {
    mClip = 0;
    mTime = 0.0f;
    SetSkeleton(skeleton);
}
```

1.  实现`SetSkeleton`函数，将提供的骨骼复制到`CrossFadeController`中。它标记该类的骨骼已设置，并将静止姿势复制到交叉淡出控制器的内部姿势中：

```cpp
void CrossFadeController::SetSkeleton(
                          Skeleton& skeleton) {
    mSkeleton = skeleton;
    mPose = mSkeleton.GetRestPose();
    mWasSkeletonSet = true;
}
```

1.  实现`Play`函数。此函数应清除任何活动的交叉淡出。它应设置剪辑和播放时间，但还需要将当前姿势重置为骨骼的静止姿势：

```cpp
void CrossFadeController::Play(Clip* target) {
    mTargets.clear();
    mClip = target;
    mPose = mSkeleton.GetRestPose();
    mTime = target->GetStartTime();
}
```

1.  实现`FadeTo`函数，该函数应检查请求的淡出目标是否有效。淡出目标仅在不是淡出列表中的第一个或最后一个项目时才有效。假设满足这些条件，`FadeTo`函数将提供的动画剪辑和持续时间添加到淡出列表中：

```cpp
void CrossFadeController::FadeTo(Clip* target, 
                                 float fadeTime) {
    if (mClip == 0) {
        Play(target);
        return;
    }
    if (mTargets.size() >= 1) {
        Clip* clip=mTargets[mTargets.size()-1].mClip;
        if (clip == target) {
            return;
        }
    }
    else {
        if (mClip == target) {
            return;
        }
    }
    mTargets.push_back(CrossFadeTarget(target, 
           mSkeleton.GetRestPose(), fadeTime));
}
```

1.  实现`Update`函数以播放活动动画并混合任何在淡出列表中的其他动画：

```cpp
void CrossFadeController::Update(float dt) {
    if (mClip == 0 || !mWasSkeletonSet) {
        return;
    }
```

1.  将当前动画设置为目标动画，并在动画淡出完成时移除淡出对象。每帧只移除一个目标。如果要移除所有已淡出的目标，请将循环改为反向：

```cpp
    unsigned int numTargets = mTargets.size();
    for (unsigned int i = 0; i < numTargets; ++i) {
        float duration = mTargets[i].mDuration;
        if (mTargets[i].mElapsed >= duration) {
            mClip = mTargets[i].mClip;
            mTime = mTargets[i].mTime;
            mPose = mTargets[i].mPose;
            mTargets.erase(mTargets.begin() + i);
            break;
        }
    }
```

1.  将淡出列表与当前动画混合。需要对当前动画和淡出列表中的所有动画进行采样：

```cpp
    numTargets = mTargets.size();
    mPose = mSkeleton.GetRestPose();
    mTime = mClip->Sample(mPose, mTime + dt);
    for (unsigned int i = 0; i < numTargets; ++i) {
        CrossFadeTarget& target = mTargets[i];
        target.mTime = target.mClip->Sample(
                     target.mPose, target.mTime + dt);
        target.mElapsed += dt;
        float t = target.mElapsed / target.mDuration;
        if (t > 1.0f) { t = 1.0f; }
        Blend(mPose, mPose, target.mPose, t, -1);
    }
}
```

1.  使用`GetCurrentPose`和`GetCurrentclip`辅助函数完成`CrossFadeController`类的实现。这些都是简单的 getter 函数：

```cpp
Pose& CrossFadeController::GetCurrentPose() {
    return mPose;
}
Clip* CrossFadeController::GetcurrentClip() {
    return mClip;
}
```

现在，您可以创建`CrossFadeController`的实例来控制动画播放，而不是手动控制正在播放的动画。`CrossFadeController`类在开始播放新动画时会自动淡出到新动画。在下一部分中，您将探索加法动画混合。

# 加法混合

加法动画用于通过添加额外的关节运动来修改动画。一个常见的例子是向左倾斜。如果有一个向左倾斜的动画，它只是简单地弯曲了角色的脊柱，它可以添加到行走动画中，以创建一个边走边倾斜的动画，奔跑动画，或者任何其他类型的动画。

并非所有动画都适合作为加法动画。加法动画通常是专门制作的。我已经在本章的示例代码中提供的`Woman.gltf`文件中添加了一个`Lean_Left`动画。这个动画是为了加法而制作的。它只弯曲了脊柱关节中的一个。

加法动画通常不是根据时间播放，而是根据其他输入播放。以向左倾斜为例——它应该由用户的操纵杆控制。操纵杆越靠近左侧，倾斜的动画就应该越进。将加法动画的播放与时间以外的其他内容同步是很常见的。

## 声明加法动画

加法混合的函数声明在`Blending.h`中。第一个函数`MakeAditivePose`在时间`0`处对加法剪辑进行采样，生成一个输出姿势。这个输出姿势是用来将两个姿势相加的参考。

`Add`函数执行两个姿势之间的加法混合过程。加法混合公式为*result pose* = *input pose* + (*additive pose – additive base pose*)。前两个参数，即输出姿势和输入姿势，可以指向同一个姿势。要应用加法姿势，需要加法姿势和加法姿势的引用：

```cpp
Pose MakeAdditivePose(Skeleton& skeleton, Clip& clip);
void Add(Pose& output, Pose& inPose, Pose& addPose, 
         Pose& additiveBasePose, int blendroot);
```

`MadeAdditivePose`辅助函数生成`Add`函数用于其第四个参数的附加基础姿势。该函数旨在在初始化时调用。在下一节中，您将实现这些函数。

## 实现附加动画

在`Blending.cpp`中实现`MakeAdditivePose`函数。该函数仅在加载时调用。它应在剪辑的开始时间对提供的剪辑进行采样。该采样的结果是附加基础姿势：

```cpp
Pose MakeAdditivePose(Skeleton& skeleton, Clip& clip) {
    Pose result = skeleton.GetRestPose();
    clip.Sample(result, clip.GetStartTime());
    return result;
}
```

附加混合的公式为*结果姿势* = *输入姿势* + (*附加姿势 - 附加基础姿势*)。减去附加基础姿势只应用于动画的第一帧和当前帧之间的附加动画增量。因此，您只能对一个骨骼进行动画，比如脊柱骨骼之一，并实现使角色向左倾斜的效果。

要实现附加混合，需要循环遍历每个姿势的关节。与常规动画混合一样，需要考虑`blendroot`参数。使用每个关节的本地变换，按照提供的公式进行操作：

```cpp
void Add(Pose& output, Pose& inPose, Pose& addPose, 
         Pose& basePose, int blendroot) {
   unsigned int numJoints = addPose.Size();
   for (int i = 0; i < numJoints; ++i) {
      Transform input = inPose.GetLocalTransform(i);
      Transform additive = addPose.GetLocalTransform(i);
      Transform additiveBase=basePose.GetLocalTransform(i);
      if (blendroot >= 0 && 
          !IsInHierarchy(addPose, blendroot, i)) {
         continue;
       }
       // outPose = inPose + (addPose - basePose)
       Transform result(input.position + 
           (additive.position - additiveBase.position),
            normalized(input.rotation * 
           (inverse(additiveBase.rotation) * 
            additive.rotation)),
            input.scale + (additive.scale - 
            additiveBase.scale)
        );
        output.SetLocalTransform(i, result);
    }
}
```

重要信息

四元数没有减法运算符。要从四元数*A*中移除四元数*B*的旋转，需要将*B*乘以*A*的逆。四元数的逆应用相反的旋转，这就是为什么四元数乘以其逆的结果是单位。

附加动画通常用于创建新的动画变体，例如，将行走动画与蹲姿混合以创建蹲行动画。所有动画都可以与蹲姿进行附加混合，以在程序中创建动画的蹲姿版本。

# 总结

在本章中，您学会了如何混合多个动画。混合动画可以混合整个层次结构或只是一个子集。您还构建了一个系统，用于管理在播放新动画时动画之间的淡入淡出。我们还介绍了附加动画，可以在给定关节角度的情况下用于创建新的运动。

本章的可下载材料中包括四个示例。`Sample00`是本书到目前为止的所有代码。`Sample01`演示了如何使用`Blend`函数，通过定时器在行走和奔跑动画之间进行混合。`Sample02`演示了交叉淡入淡出控制器的使用，通过交叉淡入淡出到随机动画。`Sample03`演示了如何使用附加动画混合。

在下一章中，您将学习逆向运动学。逆向运动学允许您根据角色的末端位置来确定角色的肢体应该弯曲的方式。想象一下将角色的脚固定在不平整的地形上。
