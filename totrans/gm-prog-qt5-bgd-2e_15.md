# 使用 Qt 进行 3D 图形

许多现代游戏都在 3D 世界中进行。图形处理单元不断进化，允许开发者创建越来越视觉上吸引人且细节丰富的世界。虽然你可以直接使用 OpenGL 或 Vulkan 来渲染 3D 对象，但这可能相当具有挑战性。幸运的是，Qt 3D 模块提供了一个使用高级 API 进行 3D 渲染的实现。在本章中，我们将学习如何使用其功能，并了解我们如何使用 Qt 创建一个 3D 游戏。

Qt 3D 不仅限于渲染。你还将学习如何处理用户输入并在 3D 游戏中实现游戏逻辑。Qt 3D 被设计成高度高效和完全可扩展的，因此它允许你向所有 Qt 3D 系统添加自己的扩展。

Qt 3D 提供了 C++ 和 QML API，功能基本相同。虽然 C++ API 允许你修改和扩展实现，但我们将使用 QML 方法，这将允许我们编写干净且声明性的代码，并使用我们在前几章中学到的技术。通过将 Qt 3D 与 QML 的力量结合起来，你将能够迅速制作出令人惊叹的游戏！

本章主要涵盖以下主题：

+   渲染 3D 对象

+   处理用户输入

+   执行动画

+   与 3D 编辑器的集成

+   使用 C++ 与 Qt 3D 一起工作

+   与 Qt Widgets 和 Qt Quick 的集成

# Qt 3D 概述

在我们看到 Qt 3D 的实际应用之前，让我们先了解其架构的重要部分。

# 实体和组件

Qt 3D 不仅仅是一个 3D 渲染工具。当它充分发展时，它可以成为一个功能齐全的游戏引擎。这得益于其原始架构。Qt 3D 引入了一套新的抽象概念，这些概念对于其任务特别有用。

你可能已经注意到，大多数 Qt API 都大量使用了继承。例如，每个小部件类型都从 `QWidget` 派生而来，而 `QWidget` 又从 `QObject` 派生。Qt 形成了庞大的类家族树，以提供通用和特殊的行为。相比之下，Qt 3D 场景的元素是使用 **组合** 而不是继承来构建的。Qt 3D 场景的一个部分称为 **实体**，并由 `Entity` 类型表示。然而，一个 `Entity` 对象本身并没有任何特定的效果或行为。你可以通过添加 **组件** 的形式向实体添加行为片段。

每个组件控制实体行为的一部分。例如，`Transform` 组件控制实体在场景中的位置，`Mesh` 组件定义其形状，而 `Material` 组件控制表面的属性。这种方法允许你仅从所需的组件中组装实体。例如，如果你需要向场景中添加光源，你可以创建一个带有 `PointLight` 组件的实体。你仍然需要选择光源的位置，因此你还需要 `Transform` 组件。然而，对于光源来说，`Mesh` 和 `Material` 组件没有意义，所以你不需要使用它们。

实体按照经典的父子关系排列，就像任何其他 QML 对象或 QObjects 一样。实体树形成了一个 Qt 3D 场景。最顶层的实体通常负责定义场景级别的配置。这些设置通过将特殊组件（如 `RenderSettings` 和 `InputSettings`）附加到顶级 `Entity` 来指定。

# Qt 3D 模块

Qt 3D 被分割成多个模块，你可以选择在项目中使用它们。可能很难看出你需要哪些模块，所以让我们看看每个模块的用途。

# 稳定模块

`Qt3DCore` 模块实现了 Qt 3D 的基本结构。它提供了 `Entity` 和 `Component` 类型，以及其他 Qt 3D 系统的基类。`Qt3DCore` 本身不实现任何行为，仅提供其他模块使用的框架。

`Qt3DRender` 模块实现了 3D 渲染，因此它是功能最丰富的模块之一。以下是它功能的一些重要部分：

+   `GeometryRenderer` 是定义实体可见形状的基本组件类型。

+   `Mesh` 组件允许你从文件中导入实体的几何形状。

+   `Material` 组件是定义实体表面可见属性的基本组件类型。

+   `SceneLoader` 组件允许你从文件中导入具有网格和材料的实体层次结构。

+   光照组件（`DirectionalLight`、`EnvironmentLight`、`PointLight` 和 `SpotLight`）允许你控制场景的照明。

+   `FrameGraph` API 提供了一种定义场景应该如何精确渲染的方法。它允许你设置相机、实现多个视口、阴影映射、自定义着色器等等。

+   `ObjectPicker` 组件允许你找出在特定窗口点位置上的哪些实体。

接下来，`Qt3DLogic` 是一个非常小的模块，它提供了 `FrameAction` 组件。这个组件允许你为实体的每一帧执行任意操作。

最后，`Qt3DInput` 模块专注于用户输入。它提供了一些组件，允许你在游戏中处理键盘和鼠标事件。`Qt3DInput` 还包含可以用于配置输入设备的类型。

# 实验模块

在撰写本文时，所有其他 Qt 3D 模块仍在 **技术预览** 中，因此它们的 API 可能不完整。未来的 Qt 版本可能会在这些模块中引入不兼容的更改，所以如果你需要修改提供的代码以使其工作，请不要感到惊讶（我们的代码已在 Qt 5.10 上进行测试）。这些模块最终将在未来稳定下来，因此你应该检查 Qt 文档以了解它们当前的状态。

如同其名，`Qt3DAnimation` 模块负责 Qt 3D 场景中的动画。它能够处理实体的 `Transform` 组件上的关键帧动画，以及混合形状和顶点混合动画。然而，在本章中，我们不会使用此模块，因为已经熟悉的 Qt Quick 动画框架对我们来说已经足够。

`Qt3DExtras` 模块提供了不是严格必要的 Qt 3D 工作组件，但对于构建简单的第一个项目非常有用。它们包括：

+   基本几何形状（如立方体、球体等）的网格生成器

+   `ExtrudedTextMesh` 组件允许你在场景中显示 3D 文本

+   许多标准材质组件，例如 `DiffuseSpecularMaterial` 和 `GoochMaterial`

此外，`Qt3DExtras` 提供了两个便利类，允许用户使用鼠标和键盘控制相机的位置：

+   `OrbitCameraController` 沿着轨道路径移动相机

+   `FirstPersonCameraController` 以第一人称游戏的方式移动相机

`Qt3DQuickExtras` 模块提供了 `Qt3DExtras::Quick::Qt3DQuickWindow` C++ 类。这是一个显示基于 QML 的 Qt 3D 场景的窗口。

最后，`Qt3DQuickScene2D` 模块提供了将 Qt Quick 项目嵌入到 Qt 3D 场景的能力，而 `QtQuick.Scene3D` QML 模块允许你将 Qt 3D 场景嵌入到 Qt Quick 应用程序中。

如你所见，Qt 3D 的功能并不仅限于渲染。你还可以用它来处理用户输入并实现实体的游戏逻辑。Qt 3D 是完全可扩展的，因此你可以使用其 C++ API 来实现自己的组件，或者修改现有的组件。然而，在本章中，我们将主要使用基于 QML 的 API。

注意，Qt 3D 对象不是 Qt Quick 项目，因此当你使用 Qt 3D 时，并非所有 Qt Quick 功能都对你开放。例如，你不能使用 `Repeater` 来实例化多个 Qt 3D 实体。然而，你仍然可以使用 Qt Quick 动画，因为它们可以处理任何 QML 对象。你也可以使用 `Scene3D` QML 类型将 Qt 3D 场景嵌入到 Qt Quick 界面中。

# 使用模块

在使用每个 Qt 3D 模块之前，你必须单独在项目文件中启用该模块。例如，以下行将启用所有当前记录的模块：

```cpp
QT += 3dcore 3drender 3dinput 3dlogic 3danimation \
      qml quick 3dquick 3dquickextras 3dquickscene2d
```

当使用 QML 时，每个模块也必须单独导入：

```cpp
import Qt3D.Core 2.10
import Qt3D.Render 2.10
import Qt3D.Extras 2.10
import Qt3D.Input 2.0
import Qt3D.Logic 2.0
import QtQuick.Scene2D 2.9
import QtQuick.Scene3D 2.0
```

你可以看到不同的 Qt 3D 模块有不同的 QML 模块版本。一些模块在 Qt 5.10 中进行了更新，并具有我们希望在代码中使用的新功能，因此你必须指定最后一个版本（2.10），以便使新的 QML 类型可用。另一方面，一些模块没有更新，所以 2.0 是唯一可用的版本。随着新 Qt 版本的发布，最新的版本将在未来发生变化。希望 Qt 文档将包含正确的导入语句。

所有 Qt 3D 模块的 C++ 类型都放置在一个命名空间中。在其他方面，Qt 命名约定适用。例如，`Entity` QML 类型对应于 `Qt3DCore` 命名空间中的 `QEntity` C++ 类。相应的包含指令是 `#include <QEntity>`。

Qt 3D 还引入了 **方面** 的概念。方面简单地说是一段可以添加到 Qt 3D 引擎中的行为。`Qt3DQuickWindow` 类包含一个内置的方面引擎，它自动启用 `QRenderAspect`、`QInputAspect` 和 `QLogicAspect` 方面，允许 Qt 3D 渲染场景、处理用户输入和执行帧动作。如果您决定使用 `Qt3DAnimation` 模块，您也应该启用 `QAnimationAspect`。您可以使用 `Qt3DWindow::registerAspect()` 方法做到这一点。其他 Qt 3D 模块不需要单独的方面。也有可能创建一个新的方面，但通常不是必要的。

# 渲染 3D 对象

Qt 3D 场景中的每个项目都由 `Entity` 类型表示。然而，并非所有实体都是可见的 3D 对象。为了使实体可见，它必须具有 **网格** 组件和 **材质** 组件。

# 网格、材质和变换

网格定义了实体的几何形状。它包含有关顶点、边和面的信息，这些信息是渲染对象所需的。所有网格组件的基类型是 `GeometryRenderer`。然而，您通常会使用其子类之一。

+   `Mesh` 从文件中导入几何数据。

+   `ConeMesh`、`CuboidMesh`、`CylinderMesh`、`PlaneMesh`、`SphereMesh` 和 `TorusMesh` 提供了对原始几何形状的访问。

+   `ExtrudedTextMesh` 根据指定的文本和字体定义实体的形状。

虽然网格定义了对象表面将被绘制的位置，但材质定义了它将如何精确地被绘制。表面最明显的属性是其颜色，但根据反射模型，可能会有各种属性，例如漫反射和镜面反射的系数。Qt 3D 提供了大量的不同材质类型：

+   `PerVertexColorMaterial` 允许您为每个顶点设置颜色属性，并渲染周围和漫反射反射。

+   `TextureMaterial` 渲染纹理并忽略光照。

+   `DiffuseSpecularMaterial` 实现了 Phong 反射模型，并允许您设置反射的周围、漫反射和镜面反射组件。

+   `GoochMaterial` 实现了 Gooch 着色模型。

+   `MetalRoughMaterial` 使用 PBR（基于物理的渲染）渲染类似金属的表面。

+   `MorphPhongMaterial` 也遵循 Phong 反射模型，但同时也支持 `Qt3DAnimation` 模块的形态动画。

可见 3D 对象的第三个常见组件是 `Transform`。虽然不是严格必需的，但通常需要设置对象在场景中的位置。你可以使用 `translation` 属性来设置位置。也可以使用 `scale3D` 属性来缩放对象，该属性允许你为每个 `axis` 设置不同的缩放系数，或者使用接受单个系数并应用于所有轴的 `scale` 属性。同样，你可以使用 `rotation` 属性设置旋转四元数，或者使用 `rotationX`、`rotationY` 和 `rotationZ` 属性设置单个欧拉角。最后，你可以设置 `matrix` 属性来应用任意变换矩阵。

注意，变换不仅应用于当前实体，还应用于其所有子实体。

# 光照

一些可用的材料会考虑光照。Qt 3D 允许你向场景添加不同类型的灯光并对其进行配置。你可以通过向场景添加一个新的 `Entity` 并将其与一个 `DirectionalLight`、`PointLight` 或 `SpotLight` 组件相关联来实现这一点。这些组件中的每一个都有一个 `color` 属性，允许你配置灯光的颜色，以及一个 `intensity` 属性，允许你选择灯光的亮度。其余的属性都是特定于灯光类型的。

**方向光**（也称为“远光”或“日光”）从由 `DirectionalLight` 类型的 `worldDirection` 属性定义的方向发射平行光线。实体的位置和旋转对方向光的光照效果没有影响，因此不需要 `Transform` 组件。

**点光**从其位置向所有方向发射光线。可以通过附加到同一实体的 `Transform` 组件来更改光源的位置。`PointLight` 组件允许你通过设置 `constantAttenuation`、`linearAttenuation` 和 `quadraticAttenuation` 属性来配置在距离处灯光的亮度。

当点光可以解释为一个光源的球体时，**聚光灯**是一个光锥。它从其位置发射光线，但方向受到 `localDirection` 属性的限制，该属性定义了聚光灯面向的方向，以及 `cutOffAngle` 属性配置了光锥的宽度。聚光灯的位置和方向可以通过附加到同一实体的 `Transform` 组件的平移和旋转来影响。`SpotLight` 也具有与 `PointLight` 相同的衰减属性。

如果场景中没有灯光，Qt 将自动添加一个隐含的点光源，以便场景在一定程度上可见。

第四种光与其他不同。它被称为**环境光**，可以通过向实体添加`EnvironmentLight`组件来配置。它使用分配给其`irradiance`和`specular`属性的两种纹理来定义场景的周围照明。与其他光类型不同，此组件没有`color`或`intensity`属性。场景中只能有一个环境光。

注意，光源本身是不可见的。它们唯一的作用是影响使用特定材质类型的 3D 对象的外观。

# 行动时间 - 创建 3D 场景

在本章中，我们将创建著名汉诺塔游戏的实现。这个谜题游戏包含三个杆和多个不同大小的盘子。盘子可以滑到杆上，但盘子不能放在比它小的盘子上面。在起始位置，所有杆都放在一个盘子上。目标是将它们全部移动到另一个杆上。玩家一次只能移动一个盘子。

如往常一样，你将在书中附带资源中找到完整的项目。

创建一个新的 Qt Quick 应用程序 - 空项目，并将其命名为`hanoi`。虽然我们将使用一些 Qt Quick 工具，但我们的项目实际上不会基于 Qt Quick。Qt 3D 将做大部分工作。尽管如此，`Qt Quick Application - Empty`是目前最合适的模板，所以我们选择使用它。编辑`hanoi.pro`文件以启用我们将需要的 Qt 模块：

```cpp
QT += 3dcore 3drender 3dinput quick 3dquickextras
```

我们将使用`Qt3DQuickWindow`类来实例化我们的 QML 对象，而不是我们通常与 Qt Quick 一起使用的`QQmlApplicationEngine`类。为此，将`main.cpp`文件替换为以下代码：

```cpp
#include <QGuiApplication>
#include <Qt3DQuickWindow>

int main(int argc, char* argv[])
{
    QGuiApplication app(argc, argv);
    Qt3DExtras::Quick::Qt3DQuickWindow window;
    window.setSource(QUrl("qrc:/main.qml"));
    window.show();
    return app.exec();
}
```

接下来，将`main.qml`文件替换为以下代码：

```cpp
import Qt3D.Core 2.10
import Qt3D.Render 2.10
import Qt3D.Input 2.0
import Qt3D.Extras 2.10

Entity {
    components: [
        RenderSettings {
            activeFrameGraph: ForwardRenderer {
                clearColor: "black"
                camera: Camera {
                    id: camera
                    projectionType: CameraLens.PerspectiveProjection
                    fieldOfView: 45
                    nearPlane : 0.1
                    farPlane : 1000.0
                    position: Qt.vector3d(0.0, 40.0, -40.0)
                    upVector: Qt.vector3d(0.0, 1.0, 0.0)
                    viewCenter: Qt.vector3d(0.0, 0.0, 0.0)
                }
            }
        },
        InputSettings {}
    ]
}
```

此代码声明了一个包含两个组件的单个`Entity`对象。`RenderSettings`组件定义了 Qt 3D 应该如何渲染场景。`RenderSettings`的`activeFrameGraph`属性可以包含一个渲染操作的树，但最简单的帧图是一个单独的`ForwardRenderer`对象，它负责所有渲染。`ForwardRenderer`逐个将对象直接渲染到 OpenGL 帧缓冲区。我们使用`clearColor`属性将场景的背景色设置为黑色。`ForwardRenderer`的`camera`属性包含它将用于计算变换矩阵的`Camera`对象。让我们来看看我们代码中使用的`Camera`对象的属性：

+   `projectionType`属性定义了投影的类型。除了`PerspectiveProjection`之外，你还可以使用`OrthographicProjection`、`FrustumProjection`或`CustomProjection`。

+   `fieldOfView`属性包含透视投影的视野参数。你可以更改它以实现缩放效果。

+   `nearPlane`和`farPlane`属性定义了在相机中可见的最近和最远平面的位置（它们对应于视口坐标中可见的*z*轴值）。

+   `position`向量定义了相机在世界坐标中的位置。

+   世界坐标中的`upVector`向量是当通过相机观察时指向向上的向量。

+   世界坐标中的`viewCenter`向量是将在视口中心出现的点。

当使用透视投影时，通常需要根据窗口大小设置纵横比。`Camera`对象有`aspectRatio`属性用于此目的，但我们不需要设置它，因为`Qt3DQuickWindow`对象将自动更新此属性。

你可以通过在`main.cpp`文件中添加`window.setCameraAspectRatioMode(Qt3DExtras::Quick::Qt3DQuickWindow::UserAspectRatio)`来禁用此功能。

如果你想使用正交投影而不是透视投影，你可以使用`Camera`对象的`top`、`left`、`bottom`和`right`属性来设置可见区域。

最后，我们的`Entity`的第二个组件是`InputSettings`组件。它的`eventSource`属性应指向提供输入事件的对象。与`aspectRatio`一样，我们不需要手动设置此属性。`Qt3DQuickWindow`将找到`InputSettings`对象并将其自身设置为`eventSource`。

你可以运行项目以验证它是否成功编译并且没有产生任何运行时错误。你应该得到一个空的黑窗口作为结果。

现在让我们在我们的场景中添加一些可见的内容。编辑`main.qml`文件，向根`Entity`添加几个子对象，如下所示：

```cpp
Entity {
    components: [
        RenderSettings { /* ... */ },
        InputSettings {}
    ]
    FirstPersonCameraController {
 camera: camera
 }
 Entity {
 components: [
 DirectionalLight {
 color: Qt.rgba(1, 1, 1)
 intensity: 0.5
 worldDirection: Qt.vector3d(0, -1, 0)
 }
 ]
 }
 Entity {
 components: [
 CuboidMesh {},
 DiffuseSpecularMaterial { ambient: "#aaa"; shininess: 100; },
 Transform { scale: 10 }
 ]
 }
}
```

因此，你应该在窗口中心看到一个立方体：

![图片](img/c4641a69-bed6-448d-8a8e-5091b7237f54.png)

更重要的是，你可以使用箭头键、**Page Up**和**Page Down**键以及左鼠标按钮来移动相机。

# 刚才发生了什么？

我们在我们的场景图中添加了一些对象。首先，`FirstPersonCameraController`对象允许用户自由控制相机。这在还没有自己的相机控制代码时测试游戏非常有用。接下来，一个带有单个`DirectionalLight`组件的实体在场景中充当光源。我们使用该组件的属性来设置光的颜色、强度和方向。

最后，我们添加了一个代表常规 3D 对象的实体。其形状由`CuboidMesh`组件提供，该组件生成一个单位立方体。其表面的外观由符合广泛使用的 Phong 反射模型的`DiffuseSpecularMaterial`组件定义。您可以使用`ambient`、`diffuse`和`specular`颜色属性来控制反射光的不同组成部分。`shininess`属性定义了表面有多光滑。我们使用`Transform`组件将立方体缩放到更大的尺寸。

# 是时候构建汉诺塔场景了。

我们接下来的任务是为我们的谜题创建一个基础和三个杆。我们将利用 QML 的模块化系统，将我们的代码拆分成多个组件。首先，让我们将相机和光照设置保留在`main.qml`中，并将我们的实际场景内容放入一个新的`Scene`组件中。为了做到这一点，将文本光标置于立方体的实体声明上，按*Alt* + *Enter*并选择将组件移动到单独的文件中。输入`Scene`作为组件名称并确认操作。Qt Creator 将创建一个新的`Scene.qml`文件并将其添加到项目的资源中。现在`main.qml`中只包含我们场景组件的一个实例化：

```cpp
Entity {
    //...
    Scene { }
}
```

实际的实体属性被移动到了`Scene.qml`文件中。让我们调整成以下形式：

```cpp
Entity {
    id: sceneRoot
    Entity {
        components: [
            DiffuseSpecularMaterial {
                ambient: "#444"
            },
            CuboidMesh {},
            Transform {
                scale3D: Qt.vector3d(40, 1, 40)
            }
        ]
    }
}
```

我们的场景将包含多个项目，因此我们引入了一个新的`Entity`对象，并将其命名为`sceneRoot`。这个实体没有任何组件，因此它不会对场景产生任何可见的影响。这类似于`Item`类型的对象通常作为 Qt Quick 项的容器，而不提供任何视觉内容。

现在立方体实体是`sceneRoot`的子实体。我们使用`Transform`组件的`scale3D`属性来改变立方体的尺寸。现在它看起来像一张桌面，将作为其余物体的基础。

现在让我们来处理杆。自然地，我们想要有一个`Rod`组件，因为它是场景的一个重复部分。在项目树中调用`qml.qrc`的上下文菜单，并选择添加新项。从 Qt 类别中选择 QML 文件（Qt Quick 2），并将文件名输入为`Rod`。让我们看看我们如何实现这个组件：

```cpp
import Qt3D.Core 2.10
import Qt3D.Render 2.10
import Qt3D.Extras 2.10
Entity {
    property int index
    components: [
        CylinderMesh {
            id: mesh
            radius: 0.5
            length: 9
            slices: 30
        },
        DiffuseSpecularMaterial {
            ambient: "#111"
        },
        Transform {
            id: transform
            translation: {
                var radius = 8;
                var step = 2 * Math.PI / 3;
                return Qt.vector3d(radius * Math.cos(index * step),
                                   mesh.length / 2 + 0.5,
                                   radius * Math.sin(index * step));

            }
        }
    ]
}
```

与立方体实体类似，我们的杆由一个网格、一个材质和一个`Transform`组件组成。我们使用`CylinderMesh`组件来创建一个圆柱体，而不是`CubeMesh`。`radius`和`length`属性定义了对象的尺寸，而`slices`属性影响生成的三角形的数量。我们选择增加切片的数量以使圆柱体看起来更好，但请注意，这可能会对性能产生影响，如果有很多对象，这种影响可能会变得明显。

我们的`Rod`组件有一个索引属性，其中包含杆的位置编号。我们使用这个属性来计算杆的*x*和*z*坐标，以便所有三根杆都放置在一个半径为八的圆上。*y*坐标被设置为确保杆位于基础之上。我们将计算出的位置向量分配给`Transform`组件的`translation`属性。最后，将三个`Rod`对象添加到`Scene.qml`文件中：

```cpp
Entity {
    id: sceneRoot
    //...
    Rod { index: 0 }
    Rod { index: 1 }
    Rod { index: 2 }
}
```

当你运行项目时，你应该看到基础和杆：

![](img/d5089464-00d9-4aff-a550-47bd14caa50d.png)

# 现在是时候重复 3D 对象了

我们的代码是可行的，但我们创建杆的方式并不理想。首先，在`Scene.qml`中枚举杆及其索引是不方便且容易出错的。其次，我们需要有一种方法可以通过索引访问`Rod`对象，而当前的方法不允许这样做。在前几章中，我们使用`Repeater` QML 类型处理重复的 Qt Quick 对象。然而，`Repeater`不适用于`Entity`对象。它只能处理继承自 Qt Quick `Item`的类型。

我们问题的解决方案对你来说已经很熟悉了，因为第十二章*，Qt Quick 中的自定义*。我们可以使用命令式 JavaScript 代码创建 QML 对象。从`Scene.qml`文件中删除`Rod`对象，并添加以下内容：

```cpp
Entity {
    id: sceneRoot
 property variant rods: []
    Entity { /* ... */}
 Component.onCompleted: {
 var rodComponent = Qt.createComponent("Rod.qml");
 if(rodComponent.status !== Component.Ready) {
 console.log(rodComponent.errorString());
 return;
 }
 for(var i = 0; i < 3; i++) {
 var rod = rodComponent.createObject(sceneRoot, { index: i });
 rods.push(rod);
 }
 }
}
```

# 刚才发生了什么？

首先，我们创建了一个名为`rods`的属性，它将保存创建的`Rod`对象数组。接下来，我们使用`Component.onCompleted`附加属性在 QML 引擎实例化我们的根对象后运行一些 JavaScript 代码。我们的第一个动作是加载`Rod`组件并检查它是否成功加载。在获得一个功能组件对象后，我们使用其`createObject()`方法创建了三根新杆。我们使用该函数的参数传递根对象和`index`属性的值。最后，我们将`Rod`对象推入数组中。

# 现在是创建磁盘的时候了

我们接下来的任务是创建八个将滑入杆中的磁盘。我们将以处理杆类似的方式来做这件事。首先，为我们的新组件创建一个名为`Disk.qml`的新文件。将以下内容放入该文件中：

```cpp
import Qt3D.Core 2.10
import Qt3D.Render 2.10
import Qt3D.Extras 2.10

Entity {
    property int index
    property alias pos: transform.translation
    components: [
        DiffuseSpecularMaterial {
            ambient: Qt.hsla(index / 8, 1, 0.5)
        },
        TorusMesh {
            minorRadius: 1.1
            radius: 2.5 + 1 * index
            rings: 80
        },
        Transform {
            id: transform
            rotationX: 90
            scale: 0.45
        }
    ]
}
```

与杆一样，磁盘通过其索引来识别。在这种情况下，索引影响磁盘的颜色和大小。我们使用`Qt.hsla()`函数来计算磁盘的颜色，该函数接受色调、饱和度和亮度值，并返回一个可以分配给材料的`ambient`属性的`color`值。这个公式将给我们八个不同色调的有色磁盘。

磁盘的位置由`Transform`组件的`translation`属性定义。我们希望能够从外部读取和更改磁盘的位置，因此我们设置了一个名为`pos`的属性别名，以暴露`transform.translation`属性值。

接下来，我们使用`TorusMesh`组件来定义我们磁盘的形状。在现实中，环形状并不适合玩汉诺塔游戏，但暂时只能这样。在本章的后面部分，我们将用更合适的形状来替换它。`TorusMesh`组件的属性允许我们调整其一些测量值，但我们也必须对该对象应用旋转和缩放，以实现所需的大小和位置。

与将所有磁盘对象放入单个数组不同，让我们为每根杆创建一个数组。当我们把一个磁盘从一个杆移动到另一个杆时，我们将从第一个杆的数组中移除该磁盘，并将其添加到第二个杆的数组中。我们可以通过向`Rod`组件添加一个属性来实现这一点。在此过程中，我们还应该将杆的位置暴露给外部。我们需要它来定位杆上的磁盘。在`Rod.qml`中的顶级`Entity`中声明以下属性：

```cpp
readonly property alias pos: transform.translation
property var disks: []
```

`pos`属性将遵循`Transform`组件的`translation`属性的值。由于这个值是基于`index`属性计算的，我们将`pos`属性声明为`readonly`。

接下来，我们需要调整`Scene`组件的`Component.onCompleted`处理程序。初始化`diskComponent`变量，就像我们处理`rodComponent`一样。然后你可以使用以下代码创建磁盘：

```cpp
var startingRod = rods[0];
for(i = 0; i < 8; i++) {
    var disk = diskComponent.createObject(sceneRoot, { index: i });
    disk.pos = Qt.vector3d(startingRod.pos.x, 8 - i, startingRod.pos.z);
    startingRod.disks.unshift(disk);
}
```

在创建每个磁盘后，我们根据其索引和所选杆的位置设置其位置。我们将所有磁盘累积在杆的`disks`属性中。我们选择数组中磁盘的顺序，使得最底部的磁盘在开始处，顶部的磁盘在末尾。`unshift()`函数将项目添加到数组的开始处，从而得到所需的顺序。

如果你运行项目，你应该在杆上看到所有的八个环：

![图片](img/140c44d4-992a-40ee-ae7d-dc71c062cd55.png)

我们接下来需要的下一个功能是将磁盘从一个杆移动到另一个杆的能力。然而，这是玩家做出的决定，因此我们还需要一种方式来接收用户的输入。让我们看看我们有哪些处理用户输入的选项。

# 处理用户输入

在 Qt 3D 中接收事件的第一种方式是使用 Qt GUI 功能。我们使用的`Qt3DQuickWindow`类从`QWindow`继承。这允许你子类化`Qt3DQuickWindow`并重新实现其一些虚拟函数，例如`keyPressEvent()`或`mouseMoveEvent()`。你已经在 Qt API 的这一部分中很熟悉了，因为它大致与 Qt Widgets 和 Graphics View 提供的相同。Qt 3D 在这里没有引入任何特别的东西，所以我们不会过多关注这种方法。

与 Qt Quick 类似，Qt 3D 引入了一个用于接收输入事件的更高级 API。让我们看看我们如何使用它。

# 设备

Qt 3D 专注于为它处理的每个方面提供良好的抽象。这也适用于输入。在 Qt 3D 的术语中，一个应用程序可能可以访问任意数量的**物理设备**。它们由`AbstractPhysicalDevice`类型表示。在撰写本文时，有两种内置的物理设备类型：键盘和鼠标。你可以在你的 QML 文件中通过声明`KeyboardDevice`或`MouseDevice`类型的对象来访问它们。

你可以使用设备对象的属性来配置其行为。目前只有一个这样的属性：`MouseDevice`类型有一个` sensitivity`属性，它影响鼠标移动如何转换为轴输入。

在单个应用程序中创建同一设备类型的多个对象是允许的。所有设备将处理所有接收到的输入，但你可以为不同的设备对象设置不同的属性值。

你通常不希望直接从物理设备处理事件。相反，你应该设置一个**逻辑设备**，它从物理设备接收事件并将它们转换为对应用程序有意义的动作和输入。你可以使用`LogicalDevice`类型的`actions`和`axes`属性为你的设备指定一组**动作**和**轴**，Qt 3D 将识别所描述的输入并通知你的对象。

我们将提供一些代码示例来展示如何在 Qt 3D 中处理各种类型的输入。你可以通过将其放入`hanoi`项目的`main.qml`文件中或为该目的创建一个单独的项目来测试代码。

# 键盘和鼠标按钮

动作由`Action`类型表示。一个动作可以通过按下单个键、键组合或键序列来触发。这是由`Action`类型的`inputs`属性定义的。最简单类型的输入是`ActionInput`，它对单个键做出反应。

当动作被触发时，其`active`属性将从`false`变为`true`。当相应的键或键组合被释放时，`active`将变回`false`。你可以使用通常的 QML 功能来跟踪属性的变化：

```cpp
Entity {
    //...
    KeyboardDevice { id: keyboardDevice }
    MouseDevice { id: mouseDevice }
    LogicalDevice {
        actions: [
            Action {
                inputs: ActionInput {
                    sourceDevice: keyboardDevice
                    buttons: [Qt.Key_A]
                }
                onActiveChanged: {
                    console.log("A changed: ", active);
                }
            },
            Action {
                inputs: ActionInput {
                    sourceDevice: keyboardDevice
                    buttons: [Qt.Key_B]
                }
                onActiveChanged: {
                    console.log("B changed: ", active);
                }
            },
            Action {
                inputs: ActionInput {
                    sourceDevice: mouseDevice
                    buttons: [MouseEvent.RightButton]
                }
                onActiveChanged: {
                    console.log("RMB changed: ", active);
                }
            }
        ]
    }
}
```

如你所见，键盘和鼠标按钮事件的处理方式相同。然而，它们来自不同的物理设备，所以请确保你在`ActionInput`的`sourceDevice`属性中指定了正确的设备。

你可以为`ActionInput`指定多个按钮。在这种情况下，如果指定的任何按钮被按下，动作将被触发。例如，使用以下代码来处理主*Enter*键和数字键盘上的*Enter*键：

```cpp
Action {
    inputs: ActionInput {
        sourceDevice: keyboardDevice
        buttons: [Qt.Key_Return, Qt.Key_Enter]
    }
    onActiveChanged: {
        if (active) {
            console.log("enter was pressed");
        } else {
            console.log("enter was released");
        }
    }
}
```

注意，将输入处理代码放入场景的根对象中不是必需的。你可以将其放入任何`Entity`中。同时，也可以有多个实体同时处理输入事件。

# 输入和弦

`InputChord`类型允许你在同时按下多个键时触发一个动作：

```cpp
Action {
    inputs: InputChord {
        timeout: 500
        chords: [
            ActionInput {
                sourceDevice: keyboardDevice
                buttons: [Qt.Key_Q]
            },
            ActionInput {
                sourceDevice: keyboardDevice
                buttons: [Qt.Key_W]
            },
            ActionInput {
                sourceDevice: keyboardDevice
                buttons: [Qt.Key_E]
            }
        ]
    }
    onActiveChanged: {
        console.log("changed: ", active);
    }
}
```

当在 500 毫秒内按下并保持 *Q*、*W* 和 *E* 键时，将调用 `onActiveChanged` 处理器。

# 模拟（轴）输入

**轴**在 Qt 3D 中是对模拟一维输入的抽象。轴输入的一个典型来源是游戏手柄的模拟摇杆。正如其名所示，`Axis` 只表示沿单轴的运动，因此摇杆可以表示为两个轴——一个用于垂直运动，一个用于水平运动。一个压力敏感的按钮可以表示为一个轴。轴输入产生一个范围从 -1 到 1 的 `float` 值，其中零对应于中性位置。

话虽如此，在撰写本文时，Qt 3D 中没有游戏手柄支持。有可能在未来版本中添加此功能。您还可以使用 Qt 3D 的可扩展 C++ API 来实现使用 Qt Gamepad 的游戏手柄设备。然而，最简单的解决方案是直接使用 Qt Gamepad。没有任何东西阻止您在使用 Qt 3D 的应用程序中使用 QML 或 Qt Gamepad 的 C++ API。

`Axis` 类型的 `inputs` 属性允许您指定哪些输入事件应被重定向到该轴。您可以使用 `AnalogAxisInput` 类型来访问由物理设备提供的轴数据。`MouseDevice` 提供了四个基于鼠标输入的虚拟轴。其中两个基于垂直和水平滚动。另外两个基于垂直和水平指针移动，但它们仅在按下任何鼠标按钮时才工作。

`ButtonAxisInput` 类型允许您根据按下的按钮模拟一个轴。您可以使用 `scale` 属性设置每个按钮对应的轴值。当同时按下多个按钮时，使用它们的轴值的平均值。

下面的示例展示了基于鼠标和按钮的轴：

```cpp
LogicalDevice {
    axes: [
        Axis {
            inputs: [
                AnalogAxisInput {
                    sourceDevice: mouseDevice
                    axis: MouseDevice.X
                }
            ]
            onValueChanged: {
                console.log("mouse axis value", value);
            }
        },
        Axis {
            inputs: [
                ButtonAxisInput {
                    sourceDevice: keyboardDevice
                    buttons: [Qt.Key_Left]
                    scale: -1.0
                },
                ButtonAxisInput {
                    sourceDevice: keyboardDevice
                    buttons: [Qt.Key_Right]
                    scale: 1
                }
            ]
            onValueChanged: {
                console.log("keyboard axis value", value);
            }
        }
    ]
}
```

# 对象选择器

对象选择器是一个允许实体与鼠标指针交互的组件。此组件不会直接与之前描述的输入 API 交互。例如，您不需要为此提供鼠标设备。您需要做的只是将 `ObjectPicker` 组件附加到一个也包含网格的实体上。`ObjectPicker` 的信号将通知您与该实体相关的输入事件：

| **信号** | **说明** |
| --- | --- |
| `clicked(pick)` | 对象被点击。 |
| `pressed(pick)` | 在对象上按下鼠标按钮。 |
| `released(pick)` | 在 `pressed(pick)` 触发后，鼠标按钮被释放。 |
| `moved(pick)` | 鼠标指针被移动。 |
| `entered()` | 鼠标指针进入了对象的区域。 |
| `exited()` | 鼠标指针离开了对象的区域。 |

此外，当鼠标按钮在对象上按下时，`pressed`属性将被设置为`true`，而当鼠标指针在对象区域上方时，`containsMouse`属性将被设置为`true`。您可以将更改处理程序附加到这些属性或像使用 QML 中的任何其他属性一样使用它们：

```cpp
Entity {
    components: [
        DiffuseSpecularMaterial { /* ... */ },
        TorusMesh { /* ... */ },
        ObjectPicker {
            hoverEnabled: true
            onClicked: {
                console.log("clicked");
            }
            onContainsMouseChanged: {
                console.log("contains mouse?", containsMouse);
            }
        }
    ]
}
```

根据您的场景，拾取可能是一个计算密集型任务。默认情况下，使用最简单和最有效率的选项。默认对象拾取器将仅处理鼠标按下和释放事件。您可以将`dragEnabled`属性设置为`true`以处理在`pressed(pick)`触发后的鼠标移动。您还可以将`hoverEnabled`属性设置为`true`以处理所有鼠标移动，即使鼠标按钮未按下。这些属性属于`ObjectPicker`组件，因此您可以分别为每个实体单独设置它们。

也有一些全局拾取设置，它们会影响整个窗口。这些设置存储在`RenderSettings`组件的`pickingSettings`属性中，该组件通常附加到根实体上。设置可以像这样更改：

```cpp
Entity {
    components: [
        RenderSettings {
            activeFrameGraph: ForwardRenderer { /*...*/ }
 pickingSettings.pickMethod: PickingSettings.TrianglePicking
        },
        InputSettings {}
    ]
    //...
}
```

让我们来看看可能的设置。`pickResultMode`属性定义了重叠拾取器的行为。如果设置为`PickingSettings.NearestPick`，则只有离相机最近的对象将接收到事件。如果指定为`PickingSettings.AllPicks`，则所有对象都将接收到事件。

`pickMethod`属性允许您选择拾取器如何决定鼠标指针是否与对象重叠。默认值是`PickingSettings.BoundingVolumePicking`，这意味着只考虑对象的边界框。这是一个快速但不太准确的方法。为了获得更高的精度，您可以设置`PickingSettings.TrianglePicking`方法，它考虑所有网格三角形。

最后，`faceOrientationPickingMode`属性允许您选择是否使用三角形拾取的前面、背面或两个面。

# 基于帧的输入处理

在所有之前的示例中，我们使用了属性更改信号处理程序来在逻辑设备或对象拾取器的状态发生变化时执行代码。这允许您在按钮按下或释放时执行函数。然而，有时您希望在按钮按下时执行连续动作（例如，加速对象）。通过仅对代码进行少量更改，这很容易做到。

首先，您需要给具有有趣属性的对象附加一个`id`（例如`Action`、`Axis`或`ObjectPicker`）：

```cpp
LogicalDevice {
    actions: [
        Action {
 id: myAction
            inputs: ActionInput {
                sourceDevice: keyboardDevice
                buttons: [Qt.Key_A]
            }
        }
    ]
}
```

这将允许您引用其属性。接下来，您需要使用`Qt3DLogic`模块提供的`FrameAction`组件。此组件将简单地每帧发出`triggered()`信号。您可以将它附加到任何实体并按需使用输入数据：

```cpp
Entity {
    components: [
        //...
        FrameAction {
            onTriggered: {
                console.log("A state: ", myAction.active);
            }
        }        
    ]
```

你可以使用 `FrameAction` 组件来运行任何应该每帧执行一次的代码。然而，不要忘记 QML 允许你使用属性绑定，因此你可以根据用户输入设置属性值，而无需编写任何命令式代码。

# 行动时间 - 接收鼠标输入

我们的游戏相当简单，所以玩家唯一需要做的动作是选择两根杆进行移动。让我们使用 `ObjectPicker` 来检测玩家何时点击杆。

首先，将 `RenderSettings` 对象的 `pickingSettings.pickMethod` 属性在 `main.qml` 文件中设置为 `PickingSettings.TrianglePicking`（你可以使用上一节中的代码示例）。我们的场景非常简单，三角形选择不应该太慢。此设置将大大提高选择器的准确性。

下一个更改集将针对 `Rod.qml` 文件。首先，给根实体添加一个 ID 并声明一个信号，该信号将通知外界杆已被点击：

```cpp
Entity {
    id: rod
    property int index
    readonly property alias pos: transform.translation
    property var disks: []
    signal clicked()
    //...
}
```

接下来，将 `ObjectPicker` 添加到 `components` 数组中，并在选择器报告被点击时发出公共的 `clicked()` 信号：

```cpp
Entity {
    //...
    components: [
        //...
        ObjectPicker {
            id: picker
            hoverEnabled: true
            onClicked: rod.clicked()
        }
    ]
}
```

最后，让我们给玩家一个提示，当杆与鼠标指针相交时，通过高亮显示来表明杆是可点击的：

```cpp
DiffuseSpecularMaterial {
    ambient: {
        return picker.containsMouse? "#484" : "#111";
    }
},
```

当玩家将鼠标指针放在杆上时，`picker.containsMouse` 属性将变为 `true`，QML 将自动更新材质的颜色。当你运行项目时，你应该看到这种行为。下一个任务是访问 `Scene` 组件的杆的 `clicked()` 信号。为此，你需要在代码中进行以下更改：

```cpp
Component.onCompleted: {
    //...
    var setupRod = function(i) {
        var rod = rodComponent.createObject(sceneRoot, { index: i });
        rod.clicked.connect(function() {
            rodClicked(rod);
        });
        return rod;
    }

    for(var i = 0; i < 3; i++) {
        rods.push(setupRod(i));
    }
    //...

}
function rodClicked(rod) {
    console.log("rod clicked: ", rods.indexOf(rod));
}
```

由于这些更改，每当点击杆时，游戏应该将一条消息打印到应用程序输出。

# 刚才发生了什么？

首先，我们添加了一个 `setupRod()` 辅助函数，用于创建一个新的杆并将其信号连接到新的 `rodClicked()` 函数。然后我们简单地为每个索引调用 `setupRod()` 并将杆对象累积到 `rods` 数组中。`rodClicked()` 函数将包含我们游戏逻辑的其余部分，但现在它只打印被点击杆的索引到应用程序输出。

注意，`setupRod()` 函数的内容不能直接放置在 `for` 循环的 `i` 身体中。`clicked()` 信号连接到一个捕获 `rod` 变量的闭包。在函数内部，每根杆都会连接到一个捕获相应 `Rod` 的闭包

对象。在 `for` 循环中，所有闭包都会捕获公共

`rod` 变量将保存所有闭包的最后一个 `Rod` 对象。

# 执行动画

动画对于制作一款优秀的游戏至关重要。Qt 3D 提供了一个独立的模块来执行动画，但在撰写本文时，它仍然处于实验阶段。幸运的是，Qt 已经提供了多种播放动画的方法。当使用 C++ API 时，你可以使用动画框架（我们曾在第五章，*图形视图中的动画*）中了解过）。当使用 QML 时，你可以使用 Qt Quick 提供的强大且便捷的动画系统。我们在前面的章节中已经大量使用过它，所以在这里我们将看看如何将我们的知识应用到 Qt 3D 中。

Qt Quick 动画可以应用于任何 QML 对象的几乎任何属性（严格来说，有一些属性类型它无法处理，但在这里我们不会处理这些类型）。如果你查看我们项目的 QML 文件，你会发现我们场景中的几乎所有内容都是由属性定义的。这意味着你可以动画化位置、颜色、对象的尺寸以及几乎所有其他内容。

我们当前的任务将是创建一个动画，该动画显示磁盘从杆上滑起，移动到桌子的另一端，然后沿该杆滑下。我们将动画化的属性是`pos`，它是`transform.translation`属性别名。

# 动手实践 - 动画磁盘移动

我们的动画将包含三个部分，因此需要相当多的代码。我们不是直接将所有这些代码放入`Scene`组件中，而是将其放入一个单独的组件中。创建一个名为`DiskAnimation.qml`的新文件，并填充以下代码：

```cpp
import QtQuick 2.10

SequentialAnimation {
    id: rootAnimation
    property variant target: null
    property vector3d rod1Pos
    property vector3d rod2Pos
    property int startY
    property int finalY

    property int maxY: 12

    Vector3dAnimation {
        target: rootAnimation.target
        property: "pos"
        to: Qt.vector3d(rod1Pos.x, maxY, rod1Pos.z)
        duration: 30 * (maxY - startY)

    }
    Vector3dAnimation {
        target: rootAnimation.target
        property: "pos"
        to: Qt.vector3d(rod2Pos.x, maxY, rod2Pos.z)
        duration: 400
    }
    Vector3dAnimation {
        target: rootAnimation.target
        property: "pos"
        to: Qt.vector3d(rod2Pos.x, finalY, rod2Pos.z)
        duration: 30 * (maxY - finalY)
    }
}
```

# 刚才发生了什么？

我们的动画有很多属性，因为它应该足够灵活，以处理我们需要的所有情况。首先，它应该能够动画化任何磁盘，因此我们添加了`target`属性，它将包含我们当前移动的磁盘。接下来，参与移动的杆会影响磁盘的中间和最终坐标（更具体地说，是其*x*和*z*坐标）。`rod1Pos`和`rod2Pos`属性将保存杆的坐标。`startY`和`finalY`属性定义了磁盘的起始和最终坐标。这些坐标将取决于每个杆上存储的磁盘数量。最后，`maxY`属性简单地定义了磁盘在移动过程中可以达到的最大高度。

我们所动画的属性是`vector3d`类型，因此我们需要使用能够正确插值向量所有三个分量的`Vector3dAnimation`类型。我们为动画的三个部分设置了相同的`target`和`property`。然后，我们仔细计算了每个阶段后磁盘的最终位置，并将其分配给`to`属性。不需要设置`from`属性，因为动画将自动使用磁盘的当前位置作为起始点。最后，我们计算了每一步的`duration`以确保磁盘的平稳移动。

当然，我们想立即测试新的动画。将`DiskAnimation`对象添加到`Scene`组件中，并在`Component.onCompleted`处理器的末尾初始化动画：

```cpp
DiskAnimation { id: diskAnimation }
Component.onCompleted: {
    //...
    var disk1 = rods[0].disks.pop();
    diskAnimation.rod1Pos = rods[0].pos;
    diskAnimation.rod2Pos = rods[1].pos;
    diskAnimation.startY = disk1.pos.y;
    diskAnimation.finalY = 1;
    diskAnimation.target = disk1;
    diskAnimation.start();

}
```

当你运行应用程序时，你应该看到顶部圆盘从一个杆子移动到另一个杆子。

# 行动时间 - 实现游戏逻辑

所需的大部分准备工作都已完成，现在是时候使我们的游戏功能化了。玩家应该能够通过点击一个杆子然后点击另一个杆子来进行移动。在选择了第一杆后，游戏应该记住它并以不同的颜色显示它。

首先，让我们准备`Rod`组件。我们需要它有一个新的属性，表示该杆被选为下一次移动的第一杆：

```cpp
property bool isSourceRod: false
```

使用属性绑定很容易根据`isSourceRod`值使杆子改变颜色：

```cpp
DiffuseSpecularMaterial {
    ambient: {
        if (isSourceRod) {
            return picker.containsMouse? "#f44" : "#f11";
        } else {
            return picker.containsMouse? "#484" : "#111";
        }
    }
},
```

现在，让我们将注意力转向`Scene`组件。我们需要一个包含当前所选第一杆的属性：

```cpp
Entity {
    id: sceneRoot
    property variant rods: []
 property variant sourceRod
    //...
}
```

剩下的就是实现`rodClicked()`函数。让我们分两步进行：

```cpp
function rodClicked(rod) {
    if (diskAnimation.running) { return; }
    if (rod.isSourceRod) {
        rod.isSourceRod = false;
        sourceRod = null;
    } else if (!sourceRod) {
        if (rod.disks.length > 0) {
            rod.isSourceRod = true;
            sourceRod = rod;
        } else {
            console.log("no disks on this rod");
        }
    } else {
        //...
    }
}
```

首先，我们检查移动动画是否已经在运行，如果是，则忽略事件。接下来，我们检查点击的杆子是否已经被选中。在这种情况下，我们简单地取消选中杆子。这允许玩家在意外选中错误的杆子时取消移动。

如果`sourceRod`未设置，这意味着我们处于移动的第一阶段。我们检查点击的杆子上是否有圆盘，否则移动将不可能。如果一切正常，我们设置`sourceRod`属性和杆子的`isSourceRod`属性。

函数的其余部分处理移动的第二阶段：

```cpp
var targetRod = rod;
if (targetRod.disks.length > 0 &&
    targetRod.disks[targetRod.disks.length - 1].index <
    sourceRod.disks[sourceRod.disks.length - 1].index)
{
    console.log("invalid move");
} else {
    var disk = sourceRod.disks.pop();
    targetRod.disks.push(disk);
    diskAnimation.rod1Pos = sourceRod.pos;
    diskAnimation.rod2Pos = targetRod.pos;
    diskAnimation.startY = disk.pos.y;
    diskAnimation.finalY = targetRod.disks.length;
    diskAnimation.target = disk;
    diskAnimation.start();
}
sourceRod.isSourceRod = false;
sourceRod = null;
```

在这个分支中，我们已经知道我们已经在`sourceRod`属性中存储了第一杆对象。我们将点击的杆子对象存储在`targetRod`变量中。接下来，我们检查玩家是否试图将较大的圆盘放在较小的圆盘上面。如果是这样，我们拒绝执行无效的移动。

如果一切正常，我们最终执行移动。我们使用`pop()`函数从`sourceRod.disks`数组的末尾移除圆盘。这是将被移动到另一个杆子的圆盘。我们立即将圆盘对象推入另一个杆子的`disks`数组。接下来，我们仔细设置动画的所有属性并启动它。在函数的末尾，我们清除杆子的`isSourceRod`属性和场景的`sourceRod`属性，以便玩家可以进行下一次移动。

# 尝试英雄之旅 - 提升游戏

尝试对游戏进行自己的修改。例如，你可以通过闪烁背景颜色或基础对象的颜色来通知玩家一个无效的移动。你甚至可以使用`ExtrudedTextMesh`组件在场景中添加 3D 文本。尝试使用不同的缓动模式来使动画看起来更好。

`Scene`对象的属性和函数对外部世界是可见的，但它们实际上是实现细节。您可以通过将它们放入一个内部的`QtObject`中来解决这个问题，正如我们在第十二章，“Qt Quick 的自定义”中描述的那样。

在渲染方面，Qt 3D 非常灵活。虽然它与简单的`ForwardRenderer`使用起来很直接，但如果您想要创建一个更复杂的渲染图，您也可以做到。您可以将渲染输出到多个视口，使用离屏纹理，应用自定义着色器，并创建自己的图形效果和材质。我们无法在本书中讨论所有这些可能性，但您可以通过查看 Qt 示例来了解如何实现。一些相关的示例包括 Qt3D：多视口 QML、Qt3D：阴影映射 QML 和 Qt3D：高级自定义材质 QML。

# 与 3D 建模软件的集成

`Qt3DExtras`模块提供的几何形状非常适合原型设计。正如我们所见，当您想要快速创建和测试新游戏时，这些网格生成器非常有用。然而，一个真正的游戏通常包含比球体和立方体更复杂的图形。网格通常使用专门的 3D 建模软件准备。Qt 3D 提供了从外部文件导入 3D 数据的广泛功能。

导入此类数据的第一种方式是`Mesh`组件。您只需将此组件附加到实体上，并使用`source`属性指定文件路径。从 Qt 5.10 版本开始，`Mesh`支持 OBJ、PLY、STL 和 Autodesk FBX 文件格式。

与往常一样，您可以使用真实的文件名或 Qt 资源路径。但是请注意，源属性期望一个 URL，而不是路径。正确的绝对资源路径应以`qrc:/`开头，而绝对文件路径应以`file://`开头。您还可以使用相对路径，这些路径将相对于当前 QML 文件进行解析。

如果您正在使用 OBJ 文件，`Mesh`为您提供了从`source`文件中仅加载子网格的附加选项。您可以通过在`Mesh`组件的`meshName`属性中指定子网格的名称来实现。除了确切名称外，您还可以指定一个正则表达式来加载所有匹配该表达式的子网格。

# 行动时间 – 使用 OBJ 文件处理磁盘

Qt 3D 不提供适合磁盘的合适网格，但我们可以使用 3D 建模软件制作我们想要的任何形状，然后将其用于我们的项目中。您将在本书附带资源中找到所需的 OBJ 文件。文件命名从`disk0.obj`到`disk7.obj`。如果您想使用 3D 建模软件进行练习，您可以自己准备这些文件。

在你的项目目录中创建一个名为 `obj` 的子目录，并将 OBJ 文件放在那里。在 Qt Creator 的项目树中调用 `qml.qrc` 的上下文菜单，并选择“添加现有文件”。将所有 OBJ 文件添加到项目中。为了使这些文件发挥作用，我们需要编辑 `Disk.qml` 文件。从 `Transform` 组件中移除缩放和旋转。将 `TorusMesh` 替换为 `Mesh` 并将资源路径指定为 `source` 属性：

```cpp
components: [
    DiffuseSpecularMaterial { /*...*/ },
    Mesh {
        source: "qrc:/obj/disk" + index + ".obj"
    },
    Transform {
        id: transform
    }
]
```

Qt 3D 将现在使用我们新的磁盘模型：

![](img/9ac45d17-62bc-4229-bb96-e6204b89cfdb.png)

# 加载 3D 场景

当你想要从外部文件导入单个对象的形状时，`Mesh` 组件非常有用。然而，有时你希望从单个文件中导入多个对象。例如，你可以准备一些围绕你的游戏动作的装饰，然后一次性导入它们。这就是 `SceneLoader` 组件变得有用的地方。

它可以像 `Mesh` 组件一样使用：

```cpp
Entity {
    components: [
        SceneLoader {
            source: "path/to/scene/file"
        }
    ]
}
```

然而，`SceneLoader` 不是提供其实体的形状，而是创建一个整个的 `Entity` 对象树，这些对象成为 `SceneLoader` 实体的子对象。每个新的实体都将根据文件数据提供网格、材质和变换。`SceneLoader` 使用 Assimp（Open Asset Import Library）来解析源文件，因此它支持许多常见的 3D 格式。

# 使用 C++ 与 Qt 3D 一起工作

虽然 QML 是使用 Qt 3D 的强大且便捷的方式，但有时你可能出于某些原因更愿意使用 C++ 而不是 QML。例如，如果你的项目有一个庞大的 C++ 代码库或者你的团队不熟悉 JavaScript，坚持使用 C++ 可能是正确的解决方案。如果你想用你的自定义实现扩展 Qt 3D 类，你必须使用 C++ 方法。此外，如果你处理大量对象，在 C++ 中处理它们可能比在 QML 中处理要快得多。Qt 允许你自由选择 C++ 和 QML。

Qt 3D 的 QML API 主要由没有太多变化的 C++ 类组成。这意味着到目前为止你在这个章节中看到的绝大部分代码都可以通过最小努力透明地转换为等效的 C++ 代码。当你选择不使用 QML 时，你会失去其属性绑定、语法糖以及声明自动实例化的对象树的能力。然而，只要你熟悉 Qt C++ API 的核心，你就应该不会有任何问题。你将不得不手动创建对象并将它们分配给父对象。作为属性绑定的替代，你将不得不连接到属性更改信号并手动执行所需的更新。如果你已经学习了这本书的前几章，你应该不会有任何问题。

# 行动时间 - 使用 C++ 创建 3D 场景

让我们看看我们如何仅使用 C++ 代码重新创建我们的第一个 Qt 3D 场景。我们的场景将包含一个光源、一个立方体和一个第一人称相机控制器。你可以使用 Qt 控制台应用程序模板来创建项目。不要忘记在项目文件中启用你想要在项目中使用的 Qt 3D 模块：

```cpp
QT += 3dextras
CONFIG += c++11
```

与 QML 方法相比，第一个变化是你需要使用 `Qt3DWindow` 类而不是 `Qt3DQuickWindow`。`Qt3DWindow` 类执行了在 Qt 3D 应用程序中通常需要的几个操作。它设置了一个 `QForwardRenderer`、一个相机，并初始化了处理事件所需的 `QInputSettings` 对象。你可以使用 `defaultFrameGraph()` 方法访问默认帧图。默认相机可以通过 `camera()` 方法获得。默认相机的纵横比会自动根据窗口大小更新。如果你想要设置自定义帧图，请使用 `setActiveFrameGraph()` 方法。

我们小型示例中的所有代码都将放入 `main()` 函数中。让我们逐个分析它。首先，我们初始化通常的 `QGuiApplication` 对象，创建一个 `Qt3DWindow` 对象，并将其帧图和相机的首选设置应用于它：

```cpp
int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    Qt3DExtras::Qt3DWindow window;
    window.defaultFrameGraph()->setClearColor(Qt::black);

    Qt3DRender::QCamera *camera = window.camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(0, 40.0f, -40.0f));
    camera->setViewCenter(QVector3D(0, 0, 0));
    //...
}
```

接下来，我们创建一个根实体对象，它将包含我们所有的其他实体，并为相机创建一个相机控制器：

```cpp
Qt3DCore::QEntity *rootEntity = new Qt3DCore::QEntity();
Qt3DExtras::QFirstPersonCameraController *cameraController =
    new Qt3DExtras::QFirstPersonCameraController(rootEntity);
cameraController->setCamera(camera);
```

接下来，我们设置一个灯光实体：

```cpp
Qt3DCore::QEntity *lightEntity = new Qt3DCore::QEntity(rootEntity);
Qt3DRender::QDirectionalLight *lightComponent = new Qt3DRender::QDirectionalLight();
lightComponent->setColor(Qt::white);
lightComponent->setIntensity(0.5);
lightComponent->setWorldDirection(QVector3D(0, -1, 0));
lightEntity->addComponent(lightComponent);
```

重要的是，我们需要将根实体传递给 `QEntity` 构造函数，以确保新实体将成为我们场景的一部分。要向实体添加组件，我们使用 `addComponent()` 函数。下一步是设置立方体 3D 对象：

```cpp
Qt3DCore::QEntity *cubeEntity = new Qt3DCore::QEntity(rootEntity);
Qt3DExtras::QCuboidMesh *cubeMesh = new Qt3DExtras::QCuboidMesh();
Qt3DExtras::QDiffuseSpecularMaterial *cubeMaterial =
    new Qt3DExtras::QDiffuseSpecularMaterial();
cubeMaterial->setAmbient(Qt::white);
Qt3DCore::QTransform *cubeTransform = new Qt3DCore::QTransform();
cubeTransform->setScale(10);
cubeEntity->addComponent(cubeMesh);
cubeEntity->addComponent(cubeMaterial);
cubeEntity->addComponent(cubeTransform);
```

如你所见，此代码只是创建了一些对象，并将它们的属性设置为我们在 QML 示例中使用的相同值。代码的最后几行完成了我们的设置：

```cpp
window.setRootEntity(rootEntity);
window.show();
return app.exec();
```

我们将根实体传递给窗口并在屏幕上显示它。这就完成了！Qt 3D 将以与我们的 QML 项目相同的方式渲染构建的场景。

Qt 3D 类的所有属性都配备了变更通知信号，因此你可以使用连接语句来对外部属性变更做出反应。例如，如果你使用 `Qt3DInput::QAction` 组件来接收键盘或鼠标事件，你可以使用它的 `activeChanged(bool isActive)` 信号来获取关于事件的通知。你还可以使用 C++ 动画类，如 `QPropertyAnimation`，在 3D 场景中执行动画。

# 与 Qt Widgets 和 Qt Quick 的集成

虽然 Qt 3D 是一个非常强大的模块，但有时它不足以制作一个完整的游戏或应用程序。其他 Qt 模块，如 Qt Quick 或 Qt Widgets，可能非常有帮助，例如，当你在游戏的用户界面工作时。幸运的是，Qt 提供了一些方法来共同使用不同的模块。

当涉及到 Qt Widgets 时，您最好的选择是 `QWidget::createWindowContainer()` 函数。它允许您将小部件包围在 3D 视图中，并在单个窗口中显示它们。这种方法已经在 第九章 中讨论过，即 *Qt 应用程序中的 OpenGL 和 Vulkan*，并且可以应用于 Qt 3D 而无需任何更改。

然而，在硬件加速图形的世界中，Qt Widgets 的功能仍然有限。Qt Quick 在这个领域具有更大的潜力，Qt Quick 的 QML API 与 Qt 3D 之间的协同作用可能非常强大。Qt 提供了两种方法，可以在单个应用程序中将 Qt Quick 和 Qt 3D 结合起来，而不会产生显著的性能成本。让我们更详细地看看它们。

# 将 Qt Quick UI 嵌入 3D 场景

Qt 3D 允许您使用 `Scene2D` 类型将任意 Qt Quick 项目嵌入到您的 3D 场景中。这是如何工作的？首先，您需要将您的 Qt Quick 内容放入一个新的 `Scene2D` 对象中。接下来，您需要声明一个将用作表单渲染目标的纹理。每当 Qt Quick 决定更新其虚拟视图时，`Scene2D` 对象将直接将其渲染到指定的纹理中。您只需要按您想要的方式显示此纹理。最简单的方法是将它传递给附加到您的 3D 对象之一的 `TextureMaterial` 组件。

然而，这仅仅是工作的一部分。允许用户看到您的表单是件好事，但他们也应该能够与之交互。这也由 `Scene2D` 支持！为了使其工作，您需要执行以下操作：

1.  在 `RenderSettings` 中将 `pickMethod` 设置为 `TrianglePicking`。这将允许对象选择器检索关于鼠标事件的更准确信息。

1.  将 `ObjectPicker` 组件附加到所有使用由 `Scene2D` 创建的纹理的实体。将对象选择器的 `hoverEnabled` 和 `dragEnabled` 属性设置为 `true` 是一个好主意，以便使鼠标事件按预期工作。

1.  在 `Scene2D` 对象的 `entities` 属性中指定所有这些实体。

这将允许 `Scene2D` 将鼠标事件转发到 Qt Quick 内容。不幸的是，转发键盘事件目前尚不可用。

让我们看看这个方法的示例：

```cpp
import Qt3D.Core 2.0
import Qt3D.Render 2.0
import Qt3D.Input 2.0
import Qt3D.Extras 2.10
import QtQuick 2.10
import QtQuick.Scene2D 2.9
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.0
Entity {
    components: [
        RenderSettings {
            activeFrameGraph: ForwardRenderer { /*...*/ }
            pickingSettings.pickMethod: PickingSettings.TrianglePicking
        },
        InputSettings {}
    ]
    Scene2D {
        output: RenderTargetOutput {
            attachmentPoint: RenderTargetOutput.Color0
            texture: Texture2D {
                id: texture
                width: 200
                height: 200
                format: Texture.RGBA8_UNorm
            }
        }
        entities: [cube, plane]
        Rectangle {
            color: checkBox1.checked? "#ffa0a0" : "#a0a0ff"
            width: texture.width
            height: texture.height
            ColumnLayout {
                CheckBox {
                    id: checkBox1
                    text: "Toggle color"
                }
                CheckBox {
                    id: checkBox2
                    text: "Toggle cube"
                }
                CheckBox {
                    id: checkBox3
                    checked: true
                    text: "Toggle plane"
                }
            }
        }
    }
    //...
}
```

此代码设置了一个包含 `Scene2D` 对象的 Qt 3D 场景。`Scene2D` 本身在该 3D 场景中是不可见的。我们声明了一个将接收渲染的 Qt Quick 内容的纹理。您可以根据显示内容的尺寸选择纹理的 `width` 和 `height`。

接下来，我们声明我们将在这个纹理中渲染两个实体（我们将在下一部分代码中创建它们）。最后，我们将一个 Qt Quick 项目直接放置到 `Scene2D` 对象中。确保您根据纹理的尺寸设置此 Qt Quick 项目的尺寸。在我们的示例中，我们创建了一个包含三个复选框的表单。

下一部分代码创建了两个用于显示基于 Qt Quick 的纹理的项目：

```cpp
Entity {
    id: cube
    components: [
        CuboidMesh {},
        TextureMaterial {
            texture: texture
        },
        Transform {
            scale: 10
            rotationY: checkBox2.checked ? 45 : 0
        },
        ObjectPicker {
            hoverEnabled: true
            dragEnabled: true
        }
    ]
}
Entity {
    id: plane
    components: [
        PlaneMesh {
            mirrored: true
        },
        TextureMaterial {
            texture: texture
        },
        Transform {
            translation: checkBox3.checked ? Qt.vector3d(-20, 0, 0) : Qt.vector3d(20, 0, 0)
            scale: 10
            rotationX: 90
            rotationY: 180
            rotationZ: 0
        },
        ObjectPicker {
            hoverEnabled: true
            dragEnabled: true
        }

    ]
}
```

第一个项是一个立方体，第二个项是一个平面。大部分属性只是使场景看起来好的任意值。重要的是每个项都有一个`TextureMaterial`组件，并且我们将`texture`对象传递给它。每个项还有一个允许用户与之交互的`ObjectPicker`组件。请注意，我们使用了`PlaneMesh`的`mirrored`属性来以原始（非镜像）方向显示纹理。

通常一个平面物体就足以展示你的形状。我们使用两个物体纯粹是为了演示目的。

虽然 Qt Quick 项和 Qt 3D 实体存在于不同的世界中，看起来它们不会相互交互，但它们仍然在单个 QML 文件中声明，因此你可以使用属性绑定和其他 QML 技术使所有这些项协同工作。在我们的例子中，不仅根 Qt Quick 项的背景颜色由复选框控制，3D 对象也受到复选框的影响：

![图片](img/46724776-f305-4da1-8269-01df5ad34e00.png)

# 将 Qt 3D 场景嵌入到 Qt Quick 表单中

现在我们来看看如何执行相反的任务。如果你的应用程序主要是围绕 Qt Quick 构建的，这种方法很有用。这意味着你在`main()`函数中使用`QQmlApplicationEngine`类，并且你的`main.qml`文件的根对象通常是`Window`对象。用一点 3D 动作扩展你的 Qt Quick 应用程序非常简单。

我们可以将所有代码放入`main.qml`文件中，但将其拆分更方便，因为设置 3D 场景需要相当多的代码。假设你有一个名为`My3DScene.qml`的文件，它包含 3D 场景的常规内容：

```cpp
Entity {
    components: [
        RenderSettings {
            activeFrameGraph: ForwardRenderer { /*...*/ },
        InputSettings {}
    ]
    Entity { /*...*/ }
    Entity { /*...*/ }
    //...
}
```

要将此 3D 场景添加到`main.qml`文件（或任何其他基于 Qt Quick 的 QML 文件），你应该使用可以从`QtQuick.Scene3D`模块导入的`Scene3D` QML 类型。例如，这是如何创建一个带有按钮和 3D 视图的表单的方法：

```cpp
import QtQuick 2.10
import QtQuick.Layouts 1.0
import QtQuick.Controls 1.0
import QtQuick.Window 2.0
import QtQuick.Scene3D 2.0

Window {
    visible: true
    Button {
        id: button1
        text: "button1"
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
            margins: 10
        }
    }
    Scene3D {
        focus: true
        anchors {
            top: button1.bottom
            bottom: parent.bottom
            left: parent.left
            right: parent.right
            margins: 10
        }
 aspects: ["input", "logic"]
 My3DScene {}
 }
}
```

大部分代码是 Qt Quick 表单的常规内容。`Scene3D`项执行所有魔法。根 3D 实体应直接添加到该项，或者，如我们的案例所示，以自定义组件的形式添加。`Scene3D`项设置 Qt 3D 引擎并渲染传递的场景：

![图片](img/403e934d-6ae3-423a-8e4c-aa4a8257b994.png)

如果你想要使用`Qt3DInput`或`Qt3DLogic`模块，你需要使用`Scene3D`的`aspects`属性启用相应的 3D 方面，如图所示。此外，可以使用`multisample`布尔属性启用多采样。可以使用`hoverEnabled`属性在鼠标按钮未按下时处理鼠标事件。

与`Qt3DQuickWindow`类似，`Scene3D`默认自动设置摄像机的宽高比。你可以通过将其`cameraAspectRatioMode`属性设置为`Scene3D.UserAspectRatio`来禁用它。

这种方法也可以用来在 3D 视图上显示一些 UI 控件。这将允许您利用 Qt Quick 的全部功能，使您的游戏 UI 精彩绝伦。

# 快速问答

Q1. 哪个组件可以用来旋转 3D 对象？

1.  `CuboidMesh`

1.  `RotationAnimation`

1.  `Transform`

Q2. 哪个组件最适合模拟太阳光？

1.  `DirectionalLight`

1.  `PointLight`

1.  `SpotLight`

Q3. Qt 3D 材料是什么？

1.  一个允许您从文件中加载纹理的对象。

1.  定义对象物理属性的组件。

1.  定义对象表面可见属性的组件。

# 概述

在本章中，我们学习了如何使用 Qt 创建 3D 游戏。我们看到了如何在场景中创建和定位 3D 对象，并配置相机进行渲染。接下来，我们探讨了如何使用 Qt 3D 处理用户输入。不仅如此，您还学会了将现有的动画技能应用到 Qt 3D 对象上。最后，我们发现了如何将 Qt 3D 与其他 Qt 模块一起使用。

与 Qt Quick 一样，Qt 3D 正在快速发展。在撰写本文时，一些模块仍然是实验性的。您应该期待 Qt 3D 的 API 将得到改进和扩展，因此请确保您检查 Qt 文档以获取新版本。

这本书关于使用 Qt 进行游戏编程的内容到此结束。我们向您介绍了 Qt 的一般基础知识，描述了其小部件领域，并带您进入了 Qt Quick 和 Qt 3D 的迷人世界。Widgets（包括 Graphics View）、Qt Quick 和 Qt 3D 是您在用 Qt 框架创建游戏时可以采取的主要路径。我们还向您展示了如何利用您可能拥有的任何 OpenGL 或 Vulkan 技能，通过合并这两种方法来超越 Qt 当前的功能。此时，您应该开始尝试和实验，如果您在任何时候感到迷茫或缺乏如何做某事的信息，非常有帮助的 Qt 参考手册应该是您首先指向的资源。

祝您好运，玩得开心！
