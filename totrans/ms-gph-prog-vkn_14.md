# 14

# 使用光线追踪添加动态漫反射全局照明

到目前为止，本书中的照明一直基于来自点光源的直接照明。在本章中，我们将通过添加间接照明（在视频游戏环境中通常称为全局照明）来增强照明。

这种照明类型来源于模拟光的行为。不深入量子物理和光学，我们需要考虑的信息是光在表面反射几次，直到其能量变为零。

在电影和视频游戏中，全局照明一直是照明的一个重要方面，但通常无法实时执行。

在电影中，渲染一帧通常需要几分钟（如果不是几个小时），直到全局照明被开创。视频游戏受到了这种启发，现在也包括了它在其照明中。

在本章中，我们将通过涵盖以下主题来发现如何实现实时全局照明：

+   间接照明简介

+   动态漫反射全局照明（DDGI）简介

+   实现 DDGI

每个主题都将包含子节，以便您可以扩展所提供知识。

以下图表显示了本章中的代码如何帮助间接照明：

![图 14.1 – 间接照明输出](img/B18395_14_01.jpg)

图 14.1 – 间接照明输出

在 *图 14.1* 中，场景左侧有一个点光源。我们可以看到光线从左侧的窗帘反射到地板和右侧的柱子和窗帘上，形成了绿色。

在远处的地板上，我们可以看到天空的颜色染上了墙壁。由于它的可见性提供的遮挡，对拱门的光贡献非常低。

# 技术要求

本章的代码可以在以下网址找到：[`github.com/PacktPublishing/Mastering-Graphics-Programming-with-Vulkan/tree/main/source/chapter14`](https://github.com/PacktPublishing/Mastering-Graphics-Programming-with-Vulkan/tree/main/source/chapter14)。

# 间接照明简介

回到直接和间接照明，直接照明仅显示光与物质之间的第一次相互作用，但光继续在空间中传播，有时会反射。

从渲染的角度来看，我们使用 G 缓冲区信息来计算从我们的视角可见的表面与光的第一次相互作用，但我们对我们视野之外的数据知之甚少。

以下图表显示了直接照明：

![图 14.2 – 直接照明](img/B18395_14_02.jpg)

图 14.2 – 直接照明

*图 14.2* 描述了当前的照明设置。有发光光线，这些光线与表面相互作用。光从这些表面上反射并被相机捕捉，成为像素颜色。这是一个对现象的极度简化视图，但它包含了我们需要的基本知识。

对于间接照明，仅依靠摄像机的视角是不够的，因为我们还需要计算其他光线和几何形状如何贡献并仍然影响场景中可见的部分，即使它们位于视野之外，以及可见的表面。

对于这个问题，**光线追踪**是最好的工具：它是一种查询场景空间的方法，我们可以用它来计算不同光线弹跳如何贡献到给定片段的最终值。

下面是一个显示间接照明的图表：

![图 14.3 – 间接照明](img/B18395_14_03.jpg)

图 14.3 – 间接照明

*图 14.3*显示了间接光线从表面弹跳，直到再次击中摄像机。

图中突出显示了两个光线：

+   **间接光线 0**，从隐藏表面弹跳到蓝色地板，最终进入摄像机

+   **间接光线 0**，从另一个表面弹跳，然后从红色墙上弹跳，最终进入摄像机

使用间接照明，我们想要捕捉光线从表面弹跳的现象，无论是隐藏的还是可见的。

例如，在这个设置中，红色和蓝色表面之间存在一些光线，它们将在彼此之间弹跳，使相应颜色的表面较近的部分着色。

将间接照明添加到照明中可以增强图像的真实感和视觉质量，但我们如何实现这一点呢？

在下一节中，我们将讨论我们选择实现的方法：**动态漫反射全局照明**，或**DDGI**，这主要是由 Nvidia 的研究人员开发的，但正在迅速成为 AAA 游戏中使用最广泛的一种解决方案。

# 动态漫反射全局照明（DDGI）简介

在本节中，我们将解释 DDGI 背后的算法。DDGI 基于两个主要工具：光照探针和辐照度体积：

+   **光照探针**是空间中的点，表示为球体，它们编码了光线信息

+   **辐照度体积**定义为包含三维网格的光照探针的空间，探针之间有固定的间距

当布局规则时，采样更容易，尽管我们稍后会看到一些改进放置的方法。探针使用八面体映射进行编码，这是一种将正方形映射到球体的便捷方法。在*进一步阅读*部分提供了八面体映射背后的数学链接。

DDGI 背后的核心思想是使用光线追踪动态更新探针：对于每个探针，我们将发射一些光线并计算三角形交点的辐射度。辐射度是通过引擎中动态存在的光源计算的，能够实时响应任何光线或几何形状的变化。

由于网格的分辨率相对于屏幕上的像素较低，唯一可能的光照现象就是漫反射。以下图表概述了算法，显示了着色器（绿色矩形）和纹理（黄色椭圆）之间的关系和顺序：

![图 14.4 – 算法概述](img/B18395_14_04.jpg)

图 14.4 – 算法概述

在详细查看每个步骤之前，让我们快速概述一下算法：

1.  对每个探针执行光线追踪并计算辐射度和距离。

1.  使用一些滞后更新所有探针的辐照度，同时使用计算出的辐射度。

1.  使用光线追踪过程中的距离更新所有探针的可见性数据，再次使用一些滞后。

1.  （可选）使用光线追踪距离计算每个探针的偏移位置。

1.  通过读取更新的辐照度、可见性和探针偏移来计算间接光照。

在以下小节中，我们将介绍算法的每个步骤。

## 对每个探针进行光线追踪

这是算法的第一步。对于每个需要更新的探针的每条射线，我们必须使用动态光照对场景进行光线追踪。

在光线追踪的击中着色器中，我们计算击中三角形的全局位置和法线，并执行简化的漫反射光照计算。可选的，但成本更高，我们可以读取其他辐照度探头来为光照计算添加无限次的反弹，使其看起来更加逼真。

这里特别重要的是纹理布局：每一行代表单个探针的射线。因此，如果我们每个探针有 128 条射线，我们将有一个 128 个 texels 的行，而每一列代表一个探针。

因此，具有 128 条射线和 24 个探针的配置将产生 128x24 的纹理维度。我们将光照计算作为辐射度存储在纹理的 RGB 通道中，并将击中距离存储在 Alpha 通道中。

击中距离将用于帮助处理光泄漏和计算探针偏移。

## 探针偏移

当辐照度体积被加载到世界中或其属性发生变化（如间距或位置）时，会执行探针偏移步骤。使用光线追踪步骤中的击中距离，我们可以计算探针是否直接放置在表面上，然后为其创建偏移量。

偏移量不能大于到其他探针距离的一半，这样网格仍然在网格索引和它们的位置之间保持一定的连贯性。这一步骤只执行几次（通常，大约五次是一个合适的数字），因为持续执行会导致探针无限移动，从而引起光闪烁。

一旦计算出偏移量，每个探针都将具有最终的全局位置，这极大地提高了间接光照的视觉效果。

在这里，我们可以看到计算这些偏移量后的改进：

![图 14.5 – 带有（左）和没有（右）探针偏移的全局光照](img/B18395_14_05.jpg)

图 14.5 – 带有（左）和没有（右）探针偏移的全局光照

如您所见，位于几何体内部的探针不仅不会对采样做出光照贡献，还可以创建视觉伪影。

多亏了探针偏移，我们可以将探针放置在更好的位置。

## 探针辐照度和可见性更新

现在我们有了每个探针在应用动态光照后追踪的每条射线的结果。我们如何编码这些信息？如*动态漫反射全局照明（DDGI）简介*部分所示，其中一种方法就是使用八面体映射，它将球体展开成矩形。

由于我们正在将每个探针的辐照度存储为一个 3D 体积，我们需要一个包含每个探针矩形的纹理。我们将选择创建一个包含 MxN 个探针层的一行纹理，而高度包含其他层。

例如，如果我们有一个 3x2x4 的探针网格，每一行将包含 6 个探针（3x2），最终纹理将有 4 行。我们将执行这个步骤两次，一次用于从辐照度更新辐照度，另一次用于从每个探针的距离更新可见性。

可见性对于最小化光泄漏至关重要，辐照度和可见性存储在不同的纹理中，并且可以有不同的尺寸。

有一个需要注意的事项是，为了添加对双线性过滤的支持，我们需要在每个矩形周围存储一个额外的 1 像素边框；这在这里也会更新。

着色器将读取计算出的新辐照度和距离，以及前一帧的辐照度和可见性纹理，以混合值以避免闪烁，就像体量雾使用时间重投影那样，通过简单的滞后效应来实现。

如果光照条件发生剧烈变化，滞后效应可以动态地改变，以对抗使用滞后效应的缓慢更新。结果通常对光运动的反应较慢，但这是为了避免闪烁而必须接受的缺点。

着色器的最后部分涉及更新双线性过滤的边缘。双线性过滤需要按照特定的顺序读取样本，如下面的图所示：

![图 14.6 – 双线性过滤样本。外部网格复制每个矩形内写入的像素位置](img/B18395_14_06.jpg)

图 14.6 – 双线性过滤样本。外部网格复制每个矩形内写入的像素位置

*图 14**.6* 展示了复制像素的坐标计算：中心区域是执行了完整的辐照度/可见性更新的区域，而边缘则复制指定坐标处的像素值。

我们将运行两个不同的着色器 – 一个用于更新探针辐照度，另一个用于更新探针可见性。

在着色器代码中，我们将看到实际执行此操作的代码。我们现在准备好采样探针的辐照度，如下一小节所示。

## 探针采样

这一步涉及读取辐照度探针并计算间接光照贡献。我们将从主相机的视角进行渲染，并且给定一个世界位置和方向，我们将采样最近的八个探针。可见性纹理用于最小化泄漏并软化光照结果。

由于漫反射间接组件具有软光照特性，为了获得更好的性能，我们选择在四分之一分辨率下采样，因此我们需要特别注意采样位置以避免像素不精确。

当查看探针光线追踪、辐照度更新、可见性更新、探针偏移和探针采样时，我们描述了实现 DDGI 所需的所有基本步骤。

可以包括其他步骤来使渲染更快，例如使用距离来计算非活动探针。还可以包括其他扩展，例如包含一系列体积和手动放置的体积，这些体积为 DDGI 提供了在视频游戏中使用的最佳灵活性，因为不同的硬件配置可以决定算法选择。

在下一节中，我们将学习如何实现 DDGI。

# 实现 DDGI

我们将首先读取的是光线追踪着色器。正如我们在*第十二章*中看到的，“开始使用光线追踪”，这些着色器作为一个包含光线生成、光线击中和光线丢失着色器的包提供。

这里将使用一组不同的方法将世界空间转换为网格索引，反之亦然，这些方法将在这里使用；它们包含在代码中。

首先，我们想要定义射线负载——即在光线追踪查询执行后缓存的那些信息：

```cpp
struct RayPayload {
    vec3 radiance;
    float distance;
};
```

## 光线生成着色器

第一个着色器称为光线生成。它使用球面上的随机方向和球面斐波那契序列从探针位置生成光线。

就像 TAA 和体积雾的抖动一样，使用随机方向和时间累积（在探针更新着色器中发生）可以让我们获得更多关于场景的信息，从而增强视觉效果：

```cpp
layout( location = 0 ) rayPayloadEXT RayPayload payload;
void main() {
const ivec2 pixel_coord = ivec2(gl_LaunchIDEXT.xy);
    const int probe_index = pixel_coord.y;
    const int ray_index = pixel_coord.x;
    // Convert from linear probe index to grid probe 
       indices and then position:
    ivec3 probe_grid_indices = probe_index_to_grid_indices( 
      probe_index );
    vec3 ray_origin = grid_indices_to_world( 
      probe_grid_indices probe_index );
    vec3 direction = normalize( mat3(random_rotation) * 
      spherical_fibonacci(ray_index, probe_rays) );
    payload.radiance = vec3(0);
    payload.distance = 0;
    traceRayEXT(as, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, 
      ray_origin, 0.0, direction, 100.0, 0);

    // Store the result coming from Hit or Miss shaders
    imageStore(global_images_2d[ radiance_output_index ], 
    pixel_coord, vec4(payload.radiance, payload.distance));
} 
```

## 光线击中着色器

这里是所有重负载发生的地方。

首先，我们必须声明负载和重心坐标来计算正确的三角形数据：

```cpp
layout( location = 0 ) rayPayloadInEXT RayPayload payload;
hitAttributeEXT vec2 barycentric_weights;
```

然后，检查背面三角形，只存储距离，因为不需要光照：

```cpp
void main() {
    vec3 radiance = vec3(0);
    float distance = 0.0f;
    if (gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT) {
        // Track backfacing rays with negative distance
        distance = gl_RayTminEXT + gl_HitTEXT;
        distance *= -0.2;        
    }
```

否则，计算三角形数据并执行光照：

```cpp
    else {
```

接下来，读取网格实例数据和索引缓冲区：

```cpp
    uint mesh_index = mesh_instance_draws[ 
      gl_GeometryIndexEXT ].mesh_draw_index;
    MeshDraw mesh = mesh_draws[ mesh_index ];

    int_array_type index_buffer = int_array_type( 
      mesh.index_buffer );
    int i0 = index_buffer[ gl_PrimitiveID * 3 ].v;
    int i1 = index_buffer[ gl_PrimitiveID * 3 + 1 ].v;
    int i2 = index_buffer[ gl_PrimitiveID * 3 + 2 ].v;
```

现在，我们可以从网格缓冲区读取顶点并计算世界空间位置：

```cpp
    float_array_type vertex_buffer = float_array_type( 
      mesh.position_buffer );
    vec4 p0 = vec4(vertex_buffer[ i0 * 3 + 0 ].v, 
      vertex_buffer[ i0 * 3 + 1 ].v,
      vertex_buffer[ i0 * 3 + 2 ].v, 1.0 );
    // Calculate p1 and p2 using i1 and i2 in the same 
       way.   
```

计算世界位置：

```cpp
    const mat4 transform = mesh_instance_draws[ 
      gl_GeometryIndexEXT ].model;
    vec4 p0_world = transform * p0;
    // calculate as well p1_world and p2_world
```

就像读取顶点位置一样，读取 UV 缓冲区并计算三角形的最终 UV 值：

```cpp
    float_array_type uv_buffer = float_array_type( 
      mesh.uv_buffer );
    vec2 uv0 = vec2(uv_buffer[ i0 * 2 ].v, uv_buffer[ 
      i0 * 2 + 1].v);
    // Read uv1 and uv2 using i1 and i2 
    float b = barycentric_weights.x;
    float c = barycentric_weights.y;
    float a = 1 - b - c;

    vec2 uv = ( a * uv0 + b * uv1 + c * uv2 );
```

读取漫反射纹理。我们还可以读取较低的 MIP 级别以改善性能：

```cpp
    vec3 diffuse = texture( global_textures[ 
      nonuniformEXT( mesh.textures.x ) ], uv ).rgb;
```

读取三角形法线并计算最终法线。您不需要读取法线纹理，因为缓存的计算结果非常小，这些细节已经丢失：

```cpp
    float_array_type normals_buffer = 
      float_array_type( mesh.normals_buffer );
    vec3 n0 = vec3(normals_buffer[ i0 * 3 + 0 ].v,
      normals_buffer[ i0 * 3 + 1 ].v,
      normals_buffer[ i0 * 3 + 2 ].v );
    // Similar calculations for n1 and n2 using i1 and 
       i2
    vec3 normal = a * n0 + b * n1 + c * n2;
    const mat3 normal_transform = mat3(mesh_instance_draws
      [gl_GeometryIndexEXT ].model_inverse);
    normal = normal_transform * normal;
```

我们可以计算世界位置和法线，然后计算直接光照：

```cpp
    const vec3 world_position = a * p0_world.xyz + b * 
      p1_world.xyz + c * p2_world.xyz;
    vec3 diffuse = albedo * direct_lighting(world_position, 
      normal);
    // Optional: infinite bounces by samplying previous 
       frame Irradiance:
    diffuse += albedo * sample_irradiance( world_position, 
      normal, camera_position.xyz ) * 
      infinite_bounces_multiplier;
```

最后，我们可以缓存辐射度和距离：

```cpp
    radiance = diffuse;
    distance = gl_RayTminEXT + gl_HitTEXT;
    }
```

现在，让我们将结果写入负载：

```cpp
    payload.radiance = radiance;
    payload.distance = distance;
}
```

## 光线丢失着色器

在这个着色器中，我们简单地返回天空颜色。或者，如果存在，可以添加环境立方体贴图：

```cpp
layout( location = 0 ) rayPayloadInEXT RayPayload payload;
void main() {
payload.radiance = vec3( 0.529, 0.807, 0.921 );
payload.distance = 1000.0f;
}
```

## 更新探测器的辐照度和可见性着色器

这个计算着色器将读取前一帧的辐照度/可见性和当前帧的辐射/距离，并更新每个探测器的八面体表示。这个着色器将执行两次——一次用于更新辐照度，一次用于更新可见性。它还将更新边界以支持双线性过滤。

首先，我们必须检查当前像素是否是边界。如果是，我们必须更改模式：

```cpp
layout (local_size_x = 8, local_size_y = 8, local_size_z = 
        1) in;
void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    const uint probe_with_border_side = probe_side_length + 
                                        2;
    const uint probe_last_pixel = probe_side_length + 1;
    int probe_index = get_probe_index_from_pixels
      (coords.xy, int(probe_with_border_side), 
      probe_texture_width);
    // Check if thread is a border pixel
    bool border_pixel = ((gl_GlobalInvocationID.x % 
      probe_with_border_side) == 0) || 
      ((gl_GlobalInvocationID.x % probe_with_border_side ) 
      == probe_last_pixel );
    border_pixel = border_pixel || 
      ((gl_GlobalInvocationID.y % probe_with_border_side) 
      == 0) || ((gl_GlobalInvocationID.y % 
      probe_with_border_side ) == probe_last_pixel );
```

对于非边界像素，根据射线方向和用八面体坐标编码的球体方向计算权重，并将辐照度计算为辐射的总权重：

```cpp
    if ( !border_pixel ) {
        vec4 result = vec4(0);
        uint backfaces = 0;
        uint max_backfaces = uint(probe_rays * 0.1f); 
```

添加每个射线的贡献：

```cpp
        for ( int ray_index = 0; ray_index < probe_rays; 
              ++ray_index ) {
            ivec2 sample_position = ivec2( ray_index, 
              probe_index );
            vec3 ray_direction = normalize( 
              mat3(random_rotation) * 
              spherical_fibonacci(ray_index, probe_rays) );
            vec3 texel_direction = oct_decode
              (normalized_oct_coord(coords.xy));
            float weight = max(0.0, dot(texel_direction, 
              ray_direction));
```

读取这个射线的距离，如果背面太多则提前退出：

```cpp
            float distance = texelFetch(global_textures
              [nonuniformEXT(radiance_output_index)], 
              sample_position, 
              0).w;
            if ( distance < 0.0f && 
                 use_backfacing_blending() ) {
                ++backfaces;
                // Early out: only blend ray radiance into 
                   the probe if the backface threshold 
                   hasn't been exceeded
                if (backfaces >= max_backfaces) {
                    return;
                }
                continue;
            }
```

在这一点上，根据我们是在更新辐照度还是可见性，我们将执行不同的计算。

对于**辐照度**，我们必须做以下事情：

```cpp
            if (weight >= EPSILON) {
                vec3 radiance = texelFetch(global_textures
                  [nonuniformEXT(radiance_output_index)], 
                  sample_position, 0).rgb;
                radiance.rgb *= energy_conservation;

                // Storing the sum of the weights in alpha 
                   temporarily
                result += vec4(radiance * weight, weight);
            }

```

对于**可见性**，我们必须读取并限制距离：

```cpp
            float probe_max_ray_distance = 1.0f * 1.5f;
            if (weight >= EPSILON) {
                float distance = texelFetch(global_textures
                  [nonuniformEXT(radiance_output_index)], 
                  sample_position, 0).w;
                // Limit distance
                distance = min(abs(distance), 
                  probe_max_ray_distance);
                vec3 value = vec3(distance, distance * 
                  distance, 0);
                // Storing the sum of the weights in alpha 
                   temporarily
                result += vec4(value * weight, weight);
            }
        }
```

最后，应用权重：

```cpp
        if (result.w > EPSILON) {
            result.xyz /= result.w;
            result.w = 1.0f;
        }
```

现在，我们可以读取前一帧的辐照度或可见性，并使用滞后性进行混合。

对于**辐照度**，我们必须做以下事情：

```cpp
        vec4 previous_value = imageLoad( irradiance_image, 
          coords.xy );
        result = mix( result, previous_value, hysteresis );
        imageStore(irradiance_image, coords.xy, result);
```

对于**可见性**，我们必须做以下事情：

```cpp
        vec2 previous_value = imageLoad( visibility_image, 
          coords.xy ).rg;
        result.rg = mix( result.rg, previous_value, 
          hysteresis );
        imageStore(visibility_image, coords.xy, 
          vec4(result.rg, 0, 1));
```

在这一点上，我们结束非边界像素的着色器。我们将等待局部组完成并将像素复制到边界：

```cpp
        // NOTE: returning here.
        return;
    }
```

接下来，我们必须处理边界像素。

由于我们正在处理一个与每个正方形一样大的本地线程组，当组完成时，我们可以使用当前更新的数据复制边界像素。这是一个优化过程，有助于我们避免调度其他两个着色器并添加屏障等待更新完成。

在实现前面的代码后，我们必须等待组完成：

```cpp
    groupMemoryBarrier();
    barrier();
```

一旦这些屏障在着色器代码中，所有组都将完成。

我们有最终存储在纹理中的辐照度/可见性，因此我们可以复制边界像素以添加双线性采样支持。如图*图 14**.6*所示，我们需要按特定顺序读取像素以确保双线性过滤正常工作。

首先，我们必须计算源像素坐标：

```cpp
    const uint probe_pixel_x = gl_GlobalInvocationID.x % 
      probe_with_border_side;
    const uint probe_pixel_y = gl_GlobalInvocationID.y % 
      probe_with_border_side;
    bool corner_pixel = (probe_pixel_x == 0 || 
      probe_pixel_x == probe_last_pixel) && (probe_pixel_y 
      == 0 || probe_pixel_y == probe_last_pixel);
    bool row_pixel = (probe_pixel_x > 0 && probe_pixel_x < 
      probe_last_pixel);
    ivec2 source_pixel_coordinate = coords.xy;
    if ( corner_pixel ) {
        source_pixel_coordinate.x += probe_pixel_x == 0 ? 
          probe_side_length : -probe_side_length;
        source_pixel_coordinate.y += probe_pixel_y == 0 ? 
          probe_side_length : -probe_side_length;
     }
    else if ( row_pixel ) {
        source_pixel_coordinate.x += 
          k_read_table[probe_pixel_x - 1];
        source_pixel_coordinate.y += (probe_pixel_y > 0) ? 
          -1 : 1;
     }
    else {
        source_pixel_coordinate.x += (probe_pixel_x > 0) ? 
          -1 : 1;
        source_pixel_coordinate.y += 
          k_read_table[probe_pixel_y - 1];
     }

```

接下来，我们必须将源像素复制到当前边界。

对于**辐照度**，我们必须做以下事情：

```cpp
    vec4 copied_data = imageLoad( irradiance_image, 
      source_pixel_coordinate );
    imageStore( irradiance_image, coords.xy, copied_data );
```

对于**可见性**，我们必须做以下事情：

```cpp
    vec4 copied_data = imageLoad( visibility_image, 
      source_pixel_coordinate );
    imageStore( visibility_image, coords.xy, copied_data );
}
```

现在，我们已经有了更新的辐照度和可见性，准备好被场景采样。

## 间接光照采样

这个计算着色器负责读取间接辐照度，以便它可用于照明。它使用一个名为`sample_irradiance`的实用方法，该方法也用于射线命中着色器以模拟无限反弹。

首先，让我们看看计算着色器。当使用四分之一分辨率时，遍历 2x2 像素的邻域，获取最近的深度，并保存像素索引：

```cpp
layout (local_size_x = 8, local_size_y = 8, local_size_z = 
        1) in;
void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    int resolution_divider = output_resolution_half == 1 ? 
      2 : 1;
    vec2 screen_uv = uv_nearest(coords.xy, resolution / 
      resolution_divider);

    float raw_depth = 1.0f;
    int chosen_hiresolution_sample_index = 0;
    if (output_resolution_half == 1) {
        float closer_depth = 0.f;
        for ( int i = 0; i < 4; ++i ) {
            float depth = texelFetch(global_textures
             [nonuniformEXT(depth_fullscreen_texture_index)
             ], (coords.xy) * 2 + pixel_offsets[i], 0).r;
            if ( closer_depth < depth ) {
                closer_depth = depth;
                chosen_hiresolution_sample_index = i;
            }
        }

        raw_depth = closer_depth;
    }
```

使用最近深度的缓存索引读取法线：

```cpp
    vec3 normal = vec3(0);
    if (output_resolution_half == 1) {
        vec2 encoded_normal = texelFetch(global_textures
          [nonuniformEXT(normal_texture_index)],      
          (coords.xy) * 2 + pixel_offsets
          [chosen_hiresolution_sample_index], 0).rg;
       normal = normalize(octahedral_decode(encoded_normal)
       );
    }
```

现在我们已经计算了深度和法线，我们可以收集世界位置并使用法线来采样辐照度：

```cpp
    const vec3 pixel_world_position = 
      world_position_from_depth(screen_uv, raw_depth, 
      inverse_view_projection)
    vec3 irradiance = sample_irradiance( 
      pixel_world_position, normal, camera_position.xyz );
    imageStore(global_images_2d[ indirect_output_index ], 
      coords.xy, vec4(irradiance,1));
}
```

着色器的第二部分是关于`sample_irradiance`函数，它执行实际的重负载。

它首先计算一个偏差向量，将采样移动到几何体前方一点，以帮助解决泄漏问题：

```cpp
vec3 sample_irradiance( vec3 world_position, vec3 normal, 
  vec3 camera_position ) {
    const vec3 V = normalize(camera_position.xyz – 
      world_position);
    // Bias vector to offset probe sampling based on normal 
       and view vector.
    const float minimum_distance_between_probes = 1.0f;
    vec3 bias_vector = (normal * 0.2f + V * 0.8f) * 
      (0.75f  minimum_distance_between_probes) * 
      self_shadow_bias;
    vec3 biased_world_position = world_position + 
      bias_vector;

    // Sample at world position + probe offset reduces 
       shadow leaking.
    ivec3 base_grid_indices = 
      world_to_grid_indices(biased_world_position);
    vec3 base_probe_world_position = 
      grid_indices_to_world_no_offsets( base_grid_indices 
      );
```

现在我们有了采样世界位置（加上偏差）的网格世界位置和索引。

现在，我们必须计算采样位置在单元格内的每个轴上的值：

```cpp
    // alpha is how far from the floor(currentVertex) 
       position. on [0, 1] for each axis.
    vec3 alpha = clamp((biased_world_position – 
      base_probe_world_position) , vec3(0.0f), vec3(1.0f));
```

在这一点上，我们可以采样采样点的八个相邻探头：

```cpp
    vec3  sum_irradiance = vec3(0.0f);
    float sum_weight = 0.0f;
```

对于每个探头，我们必须根据索引计算其世界空间位置：

```cpp
    // Iterate over adjacent probe cage
    for (int i = 0; i < 8; ++i) {
        // Compute the offset grid coord and clamp to the 
           probe grid boundary
        // Offset = 0 or 1 along each axis
        ivec3  offset = ivec3(i, i >> 1, i >> 2) & 
          ivec3(1);
        ivec3  probe_grid_coord = clamp(base_grid_indices + 
          offset, ivec3(0), probe_counts - ivec3(1));
        int probe_index = 
          probe_indices_to_index(probe_grid_coord);
        vec3 probe_pos = 
          grid_indices_to_world(probe_grid_coord, 
          probe_index); 
```

根据网格单元顶点计算三线性权重，以在探头之间平滑过渡：

```cpp
        vec3 trilinear = mix(1.0 - alpha, alpha, offset);
        float weight = 1.0;
```

现在，我们可以看到如何使用可见性纹理。它存储深度和深度平方值，对防止光泄漏有很大帮助。

此测试基于方差，例如方差阴影图：

```cpp
        vec3 probe_to_biased_point_direction = 
          biased_world_position - probe_pos;
        float distance_to_biased_point = 
          length(probe_to_biased_point_direction);
        probe_to_biased_point_direction *= 1.0 / 
          distance_to_biased_point;
       {
            vec2 uv = get_probe_uv
              (probe_to_biased_point_direction,
              probe_index, probe_texture_width, 
              probe_texture_height, 
              probe_side_length );
            vec2 visibility = textureLod(global_textures
            [nonuniformEXT(grid_visibility_texture_index)],
            uv, 0).rg;
            float mean_distance_to_occluder = visibility.x;
            float chebyshev_weight = 1.0;
```

检查采样探头是否处于“阴影”中，并计算 Chebyshev 权重：

```cpp
            if (distance_to_biased_point > 
                mean_distance_to_occluder) {
                float variance = abs((visibility.x * 
                  visibility.x) - visibility.y);
                const float distance_diff = 
                  distance_to_biased_point – 
                  mean_distance_to_occluder;
                chebyshev_weight = variance / (variance + 
                  (distance_diff * distance_diff));
                // Increase contrast in the weight
                chebyshev_weight = max((chebyshev_weight * 
                  chebyshev_weight * chebyshev_weight), 
                    0.0f);
            }

            // Avoid visibility weights ever going all of 
               the way to zero
           chebyshev_weight = max(0.05f, chebyshev_weight);
           weight *= chebyshev_weight;
        }
```

使用为此探头计算的权重，我们可以应用三线性偏移，读取辐照度，并计算其贡献：

```cpp
         vec2 uv = get_probe_uv(normal, probe_index, 
           probe_texture_width, probe_texture_height, 
           probe_side_length );
        vec3 probe_irradiance = 
          textureLod(global_textures
          [nonuniformEXT(grid_irradiance_output_index)],
          uv, 0).rgb;
         // Trilinear weights
        weight *= trilinear.x * trilinear.y * trilinear.z + 
          0.001f;
        sum_irradiance += weight * probe_irradiance;
        sum_weight += weight;
    }
```

在采样所有探头后，最终辐照度相应缩放并返回：

```cpp
    vec3 irradiance = 0.5f * PI * sum_irradiance / 
      sum_weight;
    return irradiance;
}
```

通过这样，我们已经完成了对辐照度采样计算着色器和实用函数的查看。

可以应用更多过滤器来平滑采样，但这是由可见性数据增强的最基本版本。

现在，让我们学习如何修改`calculate_lighting`方法以添加漫反射间接光照。

## 对`calculate_lighting`方法的修改

在我们的`lighting.h`着色器文件中，一旦完成直接光照计算，添加以下行：

```cpp
    vec3 F = fresnel_schlick_roughness(max(dot(normal, V), 
      0.0), F0, roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    vec3 indirect_irradiance = textureLod(global_textures
      [nonuniformEXT(indirect_lighting_texture_index)], 
      screen_uv, 0).rgb;
    vec3 indirect_diffuse = indirect_irradiance * 
      base_colour.rgb;
    const float ao = 1.0f;
    final_color.rgb += (kD * indirect_diffuse) * ao;
```

在这里，`base_colour`是从 G 缓冲区来的漫反射，而`final_color`是计算了所有直接光照贡献的像素颜色。

基本算法已完成，但还有最后一个着色器要查看：探头偏移着色器。它计算每个探头的世界空间偏移量，以避免探头与几何体相交。

## 探头偏移着色器

此计算着色器巧妙地使用来自射线追踪传递的每条射线距离来根据后表面和前表面计数计算偏移量。

首先，我们必须检查无效的探头索引以避免写入错误的内存：

```cpp
layout (local_size_x = 32, local_size_y = 1, local_size_z = 
        1) in;
void main() {
    ivec3 coords = ivec3(gl_GlobalInvocationID.xyz);
    // Invoke this shader for each probe
    int probe_index = coords.x;
    const int total_probes = probe_counts.x * 
      probe_counts.y * probe_counts.z;
    // Early out if index is not valid
    if (probe_index >= total_probes) {
        return;
    }
```

现在，我们必须根据已计算的射线追踪距离搜索前表面和后表面击中点：

首先，声明所有必要的变量：

```cpp
    int closest_backface_index = -1;
    float closest_backface_distance = 100000000.f;
    int closest_frontface_index = -1;
    float closest_frontface_distance = 100000000.f;
    int farthest_frontface_index = -1;
    float farthest_frontface_distance = 0;
    int backfaces_count = 0;
```

对于这个探头的每条射线，读取距离并计算它是否是前表面或后表面。我们在击中着色器中存储后表面的负距离：

```cpp
    // For each ray cache front/backfaces index and 
       distances.
    for (int ray_index = 0; ray_index < probe_rays; 
         ++ray_index) {
        ivec2 ray_tex_coord = ivec2(ray_index, 
          probe_index);
        float ray_distance = texelFetch(global_textures
          [nonuniformEXT(radiance_output_index)], 
          ray_tex_coord, 0).w;
        // Negative distance is stored for backface hits in 
           the Ray Tracing Hit shader.
        if ( ray_distance <= 0.0f ) {
            ++backfaces_count;
            // Distance is a positive value, thus negate 
               ray_distance as it is negative already if
            // we are inside this branch.
            if ( (-ray_distance) < 
                  closest_backface_distance ) {
                closest_backface_distance = ray_distance;
                closest_backface_index = ray_index;
            }
        }
        else {
            // Cache either closest or farther distance and 
               indices for this ray.
            if (ray_distance < closest_frontface_distance) 
            {
                closest_frontface_distance = ray_distance;
                closest_frontface_index = ray_index;
            } else if (ray_distance > 
                       farthest_frontface_distance) {
                farthest_frontface_distance = ray_distance;
                farthest_frontface_index = ray_index;
            }
        }
    }
```

我们知道这个探头的正面和背面索引及距离。鉴于我们逐步移动探头，读取前一帧的偏移量：

```cpp
       vec4 current_offset = vec4(0);
    // Read previous offset after the first frame.
    if ( first_frame == 0 ) {
        const int probe_counts_xy = probe_counts.x * 
          probe_counts.y;
        ivec2 probe_offset_sampling_coordinates = 
          ivec2(probe_index % probe_counts_xy, probe_index 
          / probe_counts_xy);
        current_offset.rgb = texelFetch(global_textures
          [nonuniformEXT(probe_offset_texture_index)], 
          probe_offset_sampling_coordinates, 0).rgb;
    }
```

现在，我们必须检查探测器是否可以被认为是位于几何体内部，并计算一个偏离该方向的偏移量，但在这个探测器的间距限制内，我们可以称之为“单元格”：

```cpp
    vec3 full_offset = vec3(10000.f);
    vec3 cell_offset_limit = max_probe_offset * 
      probe_spacing;
    // Check if a fourth of the rays was a backface, we can 
       assume the probe is inside a geometry.
    const bool inside_geometry = (float(backfaces_count) / 
      probe_rays) > 0.25f;
    if (inside_geometry && (closest_backface_index != -1)) 
    {
        // Calculate the backface direction.
        const vec3 closest_backface_direction = 
          closest_backface_distance * normalize( 
          mat3(random_rotation) * 
          spherical_fibonacci(closest_backface_index, 
          probe_rays) );        
```

在单元格内找到最大偏移量以移动探测器：

```cpp
        const vec3 positive_offset = (current_offset.xyz + 
          cell_offset_limit) / closest_backface_direction;
        const vec3 negative_offset = (current_offset.xyz – 
          cell_offset_limit) / closest_backface_direction;
        const vec3 maximum_offset = vec3(max
          (positive_offset.x, negative_offset.x), 
          max(positive_offset.y, negative_offset.y), 
          max(positive_offset.z, negative_offset.z));
        // Get the smallest of the offsets to scale the 
           direction
        const float direction_scale_factor = min(min
          (maximum_offset.x, maximum_offset.y),
          maximum_offset.z) - 0.001f;
        // Move the offset in the opposite direction of the 
           backface one.
        full_offset = current_offset.xyz – 
          closest_backface_direction * 
          direction_scale_factor;
    }
```

如果我们没有击中背面，我们必须稍微移动探测器，使其处于静止位置：

```cpp
    else if (closest_frontface_distance < 0.05f) {
        // In this case we have a very small hit distance.
        // Ensure that we never move through the farthest 
           frontface
        // Move minimum distance to ensure not moving on a 
           future iteration.
        const vec3 farthest_direction = min(0.2f, 
          farthest_frontface_distance) * normalize( 
          mat3(random_rotation) * 
          spherical_fibonacci(farthest_frontface_index, 
          probe_rays) );
        const vec3 closest_direction = normalize(mat3
          (random_rotation) * spherical_fibonacci
          (closest_frontface_index, probe_rays));
        // The farthest frontface may also be the closest 
           if the probe can only 
        // see one surface. If this is the case, don't move 
           the probe.
        if (dot(farthest_direction, closest_direction) < 
            0.5f) {
            full_offset = current_offset.xyz + 
              farthest_direction;
        }
    } 
```

只有在偏移量在间距或单元格限制内时才更新偏移量。然后，将值存储在适当的纹理中：

```cpp
    if (all(lessThan(abs(full_offset), cell_offset_limit)))
    {
        current_offset.xyz = full_offset;
    }
    const int probe_counts_xy = probe_counts.x * 
      probe_counts.y;
    const int probe_texel_x = (probe_index % 
      probe_counts_xy);
    const int probe_texel_y = probe_index / 
      probe_counts_xy;
    imageStore(global_images_2d[ probe_offset_texture_index 
      ], ivec2(probe_texel_x, probe_texel_y), 
      current_offset);
}
```

这样，我们就计算了探测器的偏移量。

再次，这个着色器展示了如何巧妙地使用你已有的信息——在这种情况下，每条光线的探测器距离——将探测器移动到相交几何体之外。

我们展示了 DDGI 的完整功能版本，但还有一些改进可以做出，该技术可以在不同方向上扩展。一些改进的例子包括一个分类系统来禁用非贡献探测器，或者添加一个围绕相机中心具有不同网格间距的移动网格。与手动放置的体积结合，可以创建一个完整的漫反射全局照明系统。

虽然拥有具有光线追踪功能的 GPU 对于这项技术是必要的，但我们可以在静态场景部分烘焙辐照度和可见性，并在较旧的 GPU 上使用它们。另一个改进可以根据探测器的亮度变化更改滞后性，或者根据距离和重要性添加基于距离的交错探测器更新。

所有这些想法都展示了 DDGI 是多么强大和可配置，我们鼓励读者进行实验并创造其他改进。

# 摘要

在本章中，我们介绍了 DDGI 技术。我们首先讨论了全局照明，这是 DDGI 实现的照明现象。然后，我们概述了该算法，并更详细地解释了每个步骤。

最后，我们对实现中的所有着色器进行了编写和注释。DDGI 已经增强了渲染帧的照明，但它可以进一步改进和优化。

DDGI 的一个使其有用的方面是其可配置性：你可以更改辐照度和可见性纹理的分辨率，并更改光线的数量、探测器的数量和探测器的间距，以支持低端具有光线追踪功能的 GPU。

在下一章中，我们将添加另一个元素，这将帮助我们提高照明解决方案的准确性：反射！

# 进一步阅读

全局照明是一个非常大的主题，在所有渲染文献中都有广泛的覆盖，但我们想强调与 DDGI 实现更紧密相关的链接。

DDGI 本身是一个主要来自 2017 年 Nvidia 团队的想法，其核心思想在[`morgan3d.github.io/articles/2019-04-01-ddgi/index.xhtml`](https://morgan3d.github.io/articles/2019-04-01-ddgi/index.xhtml)中进行了描述。

DDGI 及其演变的原始文章如下。它们还包含了一些非常有帮助的补充代码，这些代码在实现该技术时极为有用：

+   [`casual-effects.com/research/McGuire2017LightField/index.xhtml`](https://casual-effects.com/research/McGuire2017LightField/index.xhtml)

+   [`www.jcgt.org/published/0008/02/01/`](https://www.jcgt.org/published/0008/02/01/)

+   [`jcgt.org/published/0010/02/01/`](https://jcgt.org/published/0010/02/01/)

以下是对具有球谐函数支持的 DDGI 的精彩概述，以及唯一一个用于双线性插值复制边界像素的图表。它还描述了其他有趣的主题：[`handmade.network/p/75/monter/blog/p/7288-engine_work__global_illumination_with_irradiance_probes`](https://handmade.network/p/75/monter/blog/p/7288-engine_work__global_illumination_with_irradiance_probes)。

可以在 Nvidia 的 DDGI 演示文稿中找到：[`developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9900-irradiance-fields-rtx-diffuse-global-illumination-for-local-and-cloud-graphics.pdf`](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9900-irradiance-fields-rtx-diffuse-global-illumination-for-local-and-cloud-graphics.pdf)。

以下是对全局照明的直观介绍：[`www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing`](https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing)。

*全局照明* *汇编*: [`people.cs.kuleuven.be/~philip.dutre/GI/`](https://people.cs.kuleuven.be/~philip.dutre/GI/)。

最后，这里是实时渲染的最佳网站：[`www.realtimerendering.com/`](https://www.realtimerendering.com/)。
