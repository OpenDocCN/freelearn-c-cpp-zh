# Shaders

In this chapter, we will cover the following recipes:

*   Converting GLSL shaders to SPIR-V assemblies
*   Writing vertex shaders
*   Writing tessellation control shaders
*   Writing tessellation evaluation shaders
*   Writing geometry shaders
*   Writing fragment shaders
*   Writing compute shaders
*   Writing a vertex shader that multiplies a vertex position by a projection matrix
*   Using push constants in shaders
*   Writing a texturing vertex and fragment shaders
*   Displaying polygon normals with a geometry shader

# Introduction

Most modern graphics hardware platforms render images using programmable pipeline. 3D graphics data, such as vertices and fragments/pixels, are processed in a series of steps called stages. Some stages always perform the same operations, which we can only configure to a certain extent. However, there are other stages that need to be programmed. Small programs that control the behavior of these stages are called shaders.

In Vulkan, there are five programmable graphics pipeline stages--vertex, tessellation control, evaluation, geometry, and fragment. We can also write compute shader programs for a compute pipeline. In the core Vulkan API, we control these stages with programs written in a SPIR-V. It is an intermediate language that allows us to process graphics data and perform mathematical calculation on vectors, matrices, images, buffers, or samplers. The low-level nature of this language improves compilation times. However, it also makes writing shaders harder. That's why the Vulkan SDK contains a tool called glslangValidator.

glslangValidator allows us to convert shader programs written in an OpenGL Shading Language (in short GLSL) into SPIR-V assemblies. This way, we can write shaders in a much more convenient high-level shading language, we can also easily validate them and then convert to a representation accepted by the Vulkan API, before we ship them with our Vulkan application.

In this chapter, we will learn how to write shaders using GLSL. We will see how to implement shaders for all programmable stages, how to implement tessellation or texturing, and how to use geometry shaders for debugging purposes. We will also see how to convert shaders written in a GLSL into SPIR-V assemblies using the glslangValidator program distributed with the Vulkan SDK.

# Converting GLSL shaders to SPIR-V assemblies

The Vulkan API requires us to provide shaders in the form of SPIR-V assemblies. It is a binary, intermediate representation, so writing it manually is a very hard and cumbersome task. It is much easier and quicker to write shader programs in a high-level shading language such as GLSL. After that we just need to convert them into a SPIR-V form using the glslangValidator tool.

# How to do it...

1.  Download and install the Vulkan SDK (refer to the *Downloading Vulkan SDK* recipe from [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instance and Devices*).
2.  Open the command prompt/terminal and go to the folder which contains shader files that should be converted.

3.  To convert a GLSL shader stored in the `<input>` file into a SPIR-V assembly stored in the `<output>` file, run the following command:

[PRE0]

# How it works...

The glslangValidator tool is distributed along with the Vulkan SDK. It is located in the `VulkanSDK/<version>/bin` (for 64-bit version) or `VulkanSDK/<version>/bin32` (for 32-bit version) subfolder of the SDK. It has many features, but one of its main functions is the ability to convert GLSL shaders into SPIR-V assemblies that can be consumed by the Vulkan applications.

The glslangValidator tool that converts GLSL shaders into SPIR-V assemblies is distributed with the Vulkan SDK.

The tool automatically detects the shader stage based on the extension of the `<input>` file. The available options are:

*   `vert` for the vertex shader stage
*   `tesc` for the tessellation control shader stage
*   `tese` for the tessellation evaluation shader stage
*   `geom` for the geometry shader stage
*   `frag` for the fragment shader stage
*   `comp` for the compute shader

The tool may also display the SPIR-V assembly in a readable, text form. The command presented in this recipe, stores such form in the selected `<output_txt>` file.

After GLSL shaders are converted into SPIR-V, these can be loaded in the application and used to create shader modules (refer to the *Creating a shader module* recipe from [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*).

# See also

*   In [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instances and Devices*, see the following recipe:
    *   *Downloading Vulkan SDK*
*   The following recipes in this chapter:
    *   *Writing vertex shaders*
    *   *Writing tessellation control shaders*
    *   *Writing tessellation evaluation shaders*
    *   *Writing geometry shaders*
    *   *Writing fragment shaders*
    *   *Writing compute shaders*
    *   *Creating a shader module*

# Writing vertex shaders

Vertex processing is a first graphics pipeline stage that can be programmed. Its main purpose is to convert positions of vertices, which form our geometry, from their local coordinate system into a coordinate system called a clip space. The clip coordinate system is used to allow graphics hardware to perform all following steps in a much easier and more optimal way. One of these steps is clipping, which clips processed vertices to only those that can be potentially visible, hence the name of the coordinate system. Apart from that, we can perform all the other operations, which are executed once per each vertex of drawn geometry.

# How to do it...

1.  Create a text file. Select a name for the file, but use a `vert` extension for it (for example, `shader.vert`).
2.  Insert `#version 450` in the first line of the file.
3.  Define a set of vertex input variables (attributes) that will be provided from the application for each vertex (unless otherwise specified). For each input variable:
    1.  Define its location with a location layout qualifier and an index of the attribute:
        ` layout( location = <index> )`
    2.  Provide an `in` storage qualifier
    3.  Specify the type of input variable (such as `vec4`, `float`, `int3`)
    4.  Provide a unique name of the input variable
4.  If necessary, define an output (varying) variable that will be passed (and interpolated, unless otherwise specified) to the later pipeline stages. To define each output variable:
    1.  Provide the variable's location using a location layout qualifier and an index:
        `layout( location = <index> )`
    2.  Specify an `out` storage qualifier
    3.  Specify the type of output variable (such as `vec3` or `int`)
    4.  Select a unique name of the output variable
5.  If necessary, define uniform variables that correspond to descriptor resources created in the application. To define a uniform variable:
    1.  Specify the number of descriptor set and a binding number in which a given resource can be accessed:
        `layout (set=<set index>, binding=<binding index>)`
    2.  Provide a `uniform` storage qualifier
    3.  Specify the type of the variable (such as `sampler2D`, `imageBuffer`)
    4.  Define a unique name for the variable

6.  Create a `void main()` function in which:
    1.  Perform the desired operations
    2.  Pass input variables into output variables (with or without transformations)
    3.  Store the position of the processed vertex (possibly transformed) in the `gl_Position` built-in variable.

# How it works...

The vertex processing (via the vertex shader) is the first programmable stage in a graphics pipeline. It is obligatory in every graphics pipeline that we create in Vulkan. Its main purpose is to transform positions of the vertices passed from the application from their local coordinate system into a clip space. How the transformation is done is up to us; we can omit it and provide coordinates that are already in the clip space. It is also possible for the vertex shader to do nothing at all, if later stages (tessellation or geometry shaders) calculate positions and pass them down the pipeline.

Usually though, the vertex shader takes the position provided from the application as one of the input variables (coordinates) and multiplies it (on the left side) by a model-view-projection matrix.

The main purpose of the vertex shader is to take the position of a vertex, multiply a model-view-projection matrix by it, and store the result in the `gl_Position` built-in variable.

The vertex shader can also perform other operations, pass their results to later stages of the graphics pipeline, or store them in storage images or buffers. However, we must remember that all calculations are performed once per vertex of a drawn geometry.

In the following image, a single triangle is drawn with a wireframe rendering enabled in the pipeline object. To be able to draw nonsolid geometry, we need to enable a `fillModeNonSolid` feature during the logical device creation (refer to the *Getting features and properties of a physical device* and *Creating a logical device* recipes from [Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instance and Devices*).

![](img/image_07_001.png)

To draw this triangle, a simple vertex shader was used. Here's the source code of this shader written in a GLSL:

[PRE1]

# See also

*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing a vertex shader that multiplies a vertex position by a projection matrix*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying pipeline vertex binding description, attribute description, and input state*
    *   *Creating a graphics pipeline*

# Writing tessellation control shaders

Tessellation is a process that divides geometry into much smaller parts. In graphics programming, it allows us to improve the number of details of rendered objects or to dynamically change their parameters, such as smoothness or shape, in much more flexible way.

Tessellation in Vulkan is optional. If enabled, it is performed after the vertex shader. It has three steps, of which two are programmable. The first programmable tessellation stage is used to set up parameters that control how the tessellation is performed. We do this by writing tessellation control shaders that specify values of tessellation factors.

# How to do it...

1.  Create a text file. Select a name for the file, but use a `tesc` extension for it (for example, `shader.tesc`).
2.  Insert `#version 450` in the first line of the file.
3.  Define the number of vertices that will form an output patch:

[PRE2]

4.  Define a set of input variables (attributes) that are provided from (written in) a vertex shader stage. For each input variable:
    1.  Define its location with a location layout qualifier and an index of the attribute:
        `layout( location = <index> )`
    2.  Provide an `in` storage qualifier
    3.  Specify the type of input variable (such as `vec3`, `float`)
    4.  Provide a unique name of the input variable
5.  If necessary, define an output (varying) variable that will be passed (and interpolated, unless otherwise specified) to the later pipeline stages. To define each output variable:
    1.  Provide the variable's location using a location layout qualifier and an index:
        `layout( location = <index> )`
    2.  Specify an `out` storage qualifier
    3.  Specify the type of output variable (such as `ivec2` or `bool`)
    4.  Select a unique name of the output variable
    5.  Make sure it is defined as an unsized array
6.  If necessary, define uniform variables that correspond to descriptor resources created in the application, which can be accessed in the tessellation control stage. To define a uniform variable:
    1.  Through a layout qualifier, specify the number of descriptor set and a binding number in which a given resource can be accessed:
        `layout (set=<set index>, binding=<binding index>)`
    2.  Provide a `uniform` storage qualifier.
    3.  Specify the type of the variable (such as `sampler`, `image1D`).
    4.  Define a unique name of the variable.

7.  Create a `void main()` function in which:
    1.  Perform the desired operations.
    2.  Pass input variables into output arrays of the variables (with or without transformations).
    3.  Specify the inner tessellation level factor through a `gl_TessLevelInner` variable.
    4.  Specify the outer tessellation level factor through a `gl_TessLevelOuter` variable.
    5.  Store the position of the processed patch's vertex (possibly transformed) in a `gl_out[gl_InvocationID].gl_Position` variable.

# How it works...

Tessellation shaders are optional in Vulkan; we don't have to use them. When we want to use them, we always need to use both tessellation control and tessellation evaluation shaders. We also need to enable a `tessellationShader` feature during the logical device creation.

When we want to use tessellation in our application, we need to enable a `tessellationShader` feature during the logical device creation and we need to specify both tessellation control and evaluation shader stages during the graphics pipeline creation.

The tessellation stage operates on patches. Patches are formed from vertices, but (opposed to traditional polygons) each patch may have an arbitrary number of them--from 1 to at least 32.

The tessellation control shader, as the name suggests, specifies the way in which geometry formed from the patch is tessellated. This is done through inner and outer tessellation factors that must be specified in the shader code. An inner factor, represented by the built-in `gl_TessLevelInner[]` array, specifies how the internal part of the patch is tessellated. The outer factor, which corresponds to the `gl_TessLevelOuter[]` built-in array, defines how the outer edges of the patches are tessellated. Each array element corresponds to a given edge of the patch.

The tessellation control shader is executed once for each vertex in the output patch. The index of the current vertex is available in the built-in `gl_InvocationID` variable. Only a currently processed vertex (corresponding to the current invocation) can be written to, but the shader has access to all vertices of the input patch through a `gl_in[].gl_Position` variable.

An example of a tessellation control shader that specifies arbitrary tessellation factors and passes unmodified positions may look like this:

[PRE3]

The same triangle as seen in the *Writing vertex shaders* recipe, drawn with the preceding tessellation control shader and with the tessellation evaluation shader from the *Writing tessellation evaluation shaders* recipe, should look like this:

![](img/image_07_002.png)

# See also

*   The following recipe in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing tessellation evaluation shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying pipeline tessellation state*
    *   *Creating a graphics pipeline*

# Writing tessellation evaluation shaders

Tessellation evaluation is the second programmable stage in the tessellation process. It is executed when the geometry is already tessellated (subdivided) and is used to gather results of the tessellation to form the new vertices and further modify them. When the tessellation is enabled, we need to write tessellation evaluation shaders to acquire the locations of generated vertices and provide them to the consecutive pipeline stages.

# How to do it...

1.  Create a text file. Select a name for the file and use a `tese` extension for it (for example, `shader.tese`).
2.  Insert `#version 450` in the first line of the file.
3.  Using the `in` layout qualifier, define the type of formed primitives (`isolines`, `triangles,` or `quads`), the spacing between formed vertices (`equal_spacing`, `fractional_even_spacing` or `fractional_odd_spacing`), and a winding order of generated triangles (`cw` to keep the winding provided in an application or `ccw` to reverse the winding provided in an application):

[PRE4]

4.  Define a set of input array variables that are provided from a tessellation control stage. For each input variable:
    1.  Define its location with a location layout qualifier and an index of the attribute:
        `layout( location = <index> )`
    2.  Provide an `in` storage qualifier
    3.  Specify the type of input variable (such as `vec2` or `int3`)
    4.  Provide a unique name of the input variable
    5.  Make sure it is defined as an array.

5.  If necessary, define an output (varying) variable that will be passed (and interpolated, unless otherwise specified) to later pipeline stages. To define each output variable:
    1.  Provide the variable's location using a location layout qualifier and an index:
        `layout( location = <index> )`
    2.  Specify an `out` storage qualifier
    3.  Specify the type of output variable (such as `vec4`)
    4.  Select a unique name of the output variable.
6.  If necessary, define uniform variables that correspond to descriptor resources created in the application for which the tessellation evaluation stage can have access. To define a uniform variable:
    1.  Specify the number of descriptor set and a binding number in which a given resource can be accessed:
        `layout (set=<set index>, binding=<binging index>)`
    2.  Provide a `uniform` storage qualifier
    3.  Specify the type of the variable (such as `sampler`, `image1D`)
    4.  Define a unique name of the variable.
7.  Create a `void main()` function in which:
    1.  Perform the desired operations
    2.  Use the built-in `gl_TessCoord` vector variable to generate a position of a new vertex using the positions of all the patch's vertices; modify the result to achieve the desired result, and store it in the `gl_Position` built-in variable
    3.  In a similar way, use `gl_TessCoord` to generate interpolated values of all other input variables and store them in the output variables (with additional transformations if required).

# How it works...

Tessellation control and evaluation shaders form two programmable stages required for the tessellation to work correctly. Between them is a stage that does the actual tessellation based on the parameters provided in the control stage. Results of the tessellation are acquired in the evaluation stage, where they are applied to form the new geometry.

Through tessellation evaluation we can control the way in which new primitives are aligned and formed: we specify their winding order and spacing between the generated vertices. We can also select whether we want the tessellation stage to create `isolines`, `triangles,` or `quads`.

New vertices are not created directly--the tessellator generates only barycentric tessellation coordinates for new vertices (weights), which are provided in the built-in `gl_TessCoord` variable. We can use these coordinates to interpolate between the original positions of vertices that formed a patch and place new vertices in the correct positions. That's why the evaluation shader, though executed once per generated vertex, has access to all vertices forming the patch. Their positions are provided through the `gl_Position` member of a built-in array variable, `gl_in[]`.

In case of commonly used triangles, the tessellation evaluation shader that just passes new vertices without further modifications may look like this:

[PRE5]

# See also

*   The following recipe in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing tessellation control shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying a pipeline tessellation state*
    *   *Creating a graphics pipeline*

# Writing geometry shaders

3D scenes are composed of objects called meshes. Mesh is a collection of vertices that form the external surface of an object. This surface is usually represented by triangles. When we render an object, we provide vertices and specify what type of primitives (`points`, `lines`, `triangles`) they build. After the vertices are processed by the vertex and optional tessellation stages, they are assembled into specified types of primitives. We can enable, also optional, the geometry stage and write geometry shaders that control or change the process of forming primitives from vertices. In geometry shaders, we can even create new primitives or destroy the existing ones.

# How to do it...

1.  Create a text file. Select a name for the file and use a `geom` extension for it (for example, `shader.geom`).
2.  Insert `#version 450` in the first line of the file.
3.  Using the `in` layout qualifier, define the type of primitives that are drawn in an application: `points`, `lines`, `lines_adjacency`, `triangles,` or `triangles_adjacency`:

[PRE6]

4.  Using the `out` layout qualifier, define the type of primitives that are formed (output) by the geometry shader (`points`, `line_strip` or `triangle_strip`), and the maximal number of vertices that the shader may generate:

[PRE7]

5.  Define a set of input array variables that are provided from a vertex or tessellation evaluation stage. For each input variable:
    1.  Define its location with a location layout qualifier and an index of the attribute:
        `layout( location = <index> )`
    2.  Provide an `in` storage qualifier
    3.  Specify the type of input variable (such as `ivec4`, `int` or `float`)
    4.  Provide a unique name of the input variable
    5.  Make sure the variable is defined as an unsized array

6.  If necessary, define an output (varying) variable that will be passed (and interpolated, unless otherwise specified) to the fragment shader stage. To define each output variable:
    1.  Provide variable's location using a location layout qualifier and an index:
        `layout( location = <index> )`
    2.  Specify an `out` storage qualifier
    3.  Specify the type of output variable (such as `vec3` or `uint`)
    4.  Select a unique name of the output variable
7.  If necessary, define uniform variables that correspond to descriptor resources created in the application for which the geometry stage may have access. To define a uniform variable:
    1.  Specify the number of descriptor set and a binding number in which a given resource can be accessed:
        `layout (set=<set index>, binding=<binging index>)`
    2.  Provide a `uniform` storage qualifier
    3.  Specify the type of the variable (such as `image2D`, `sampler1DArray`)
    4.  Define a unique name of the variable

8.  Create a `void main()` function in which:
    1.  Perform the desired operations
    2.  For each generated or passed vertex:
        *   Write values to output variables
        *   Store the position of the vertex (possibly transformed) in the built-in `gl_Position` variable
        *   Call `EmitVertex()` to add a vertex to the primitive
    3.  Finish the generation of the primitive by calling `EndPrimitive()` function (another primitive is implicitly started).

# How it works...

Geometry is an optional stage in a graphics pipeline. Without it, when we draw geometry, primitives are automatically generated based on the type specified during the graphics pipeline creation. Geometry shaders allow us to create additional vertices and primitives, destroy the ones drawn in an application, or to change the type of primitives formed from vertices.

The geometry shader is executed once for each primitive in a geometry drawn by the application. It has access to all vertices that constitute the primitive, or even to the adjacent ones. With this data it can pass the same or create new vertices and primitives. We must remember that we shouldn't create too many vertices in a geometry shader. If we want to create many new vertices, tessellation shaders are better suited for this task (and have a better performance). Just increasing the maximal number of vertices the geometry shader may create, even if we don't always form them; may lower the performance of our application.

We should keep the number of vertices emitted by the geometry shader as low as possible.

Geometry shaders always generate strip primitives. If we want to create separate primitives that do not form a strip, we just need to end a primitive at an appropriate moment--vertices emitted after the primitive is ended are added to the next strip so we can create as many separate strips as we choose to. Here's an example which creates three separate triangles in the original triangle's corners:

[PRE8]

When a single triangle is drawn with a simple pass-through vertex and fragment shaders, and with the preceding geometry shader, the result should look like this:

![](img/image_07_003.png)

# See also

*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Displaying polygon normals with a geometry shader*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying a pipeline input assembly state*
    *   *Creating a graphics pipeline*

# Writing fragment shaders

Fragments (or pixels) are parts of the image that can be potentially displayed on screen. They are created from geometry (drawn primitives) in a process called rasterization. They have specific screen space coordinates (x, y, and depth) but don't have any other data. We need to write a fragment shader to specify the color that needs to be displayed on screen. In the fragment shader, we can also select an attachment into which a given color should be written.

# How to do it...

1.  Create a text file. Select a name for the file, but use a `frag` extension for it (for example, `shader.frag`).
2.  Insert `#version 450` in the first line of the file.
3.  Define a set of input variables (attributes) that are provided from the earlier pipeline stages. For each input variable:
    1.  Define its location with a location layout qualifier and an index of the attribute:
        `layout( location = <index> )`
    2.  Provide an `in` storage qualifier
    3.  Specify the type of input variable (such as `vec4`, `float`, `ivec3`)
    4.  Provide a unique name of the input variable

4.  Define an output variable for each attachment into which a color should be written. To define each output variable:
    1.  Provide the variable's location (index of the attachment) using a location layout qualifier and a number:
        `layout( location = <index> )`
    2.  Specify an `out` storage qualifier
    3.  Specify the type of the output variable (such as `vec3` or `vec4`)
    4.  Select a unique name of the output variable
5.  If necessary, define uniform variables that correspond to descriptor resources created in the application. To define a uniform variable:
    1.  Specify the number of descriptor set and a binding number in which a given resource can be accessed:
        `layout (set=<set index>, binding=<binding index>)`
    2.  Provide a `uniform` storage qualifier
    3.  Specify the type of the variable (such as `sampler1D`, `subpassInput,` or `imageBuffer`)
    4.  Define a unique name of the variable
6.  Create a `void main()` function in which:
    1.  Perform the desired operations and calculations
    2.  Store the color of the processed fragment in an output variable

# How it works...

Geometry, which we draw in our application, is formed from primitives. These primitives are converted into fragments (pixels) in a process called rasterization. For each such fragment, a fragment shader is executed. Fragments may be discarded inside the shader or during framebuffer tests, such as depth, stencil, or scissor tests, so they won't even become pixels--that's why they are called fragments, not pixels.

The main purpose of a fragment shader is to set a color that will be (potentially) written to an attachment. We usually use them to perform lighting calculations and texturing. Along with compute shaders, fragment shaders are often used for post-processing effects such as bloom or deferred shading/lighting. Also, only fragment shaders can access input attachments defined in a render pass (refer to the *Creating an input attachment* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).

![](img/image_07_004.png)

To draw the triangle in the preceding figure, a simple fragment shader is used, which stores a chosen, hardcoded color:

[PRE9]

# See also

*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing a texturing vertex and fragment shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), Graphics and Compute Pipelines, see the following recipes:
    *   *Creating a shader module*
    *   *Specifying a pipeline rasterization state*
    *   *Creating a graphics pipeline*

# Writing compute shaders

Compute shaders are used, as the name suggests, for general mathematical calculations. They are executed in (local) groups of a defined, three-dimensional size, which may have access to a common set of data. At the same time, many local groups can be executed to generate results faster.

# How to do it...

1.  Create a text file. Select a name for the file, but use a `comp` extension for it (for example, `shader.comp`).
2.  Insert `#version 450` in the first line of the file.
3.  Using an input layout qualifier, define the size of the local workgroup:

[PRE10]

4.  Define uniform variables that correspond to descriptor resources created in the application. To define a uniform variable:
    1.  Specify the number of descriptor set and a binding number in which a given resource can be accessed:
        `layout (set=<set index>, binding=<binding index>)`
    2.  Provide a `uniform` storage qualifier
    3.  Specify the type of the variable (such as `image2D` or `buffer`)
    4.  Define a unique name of the variable

5.  Create a `void main()` function in which:
    1.  Perform the desired operations and calculations
    2.  Store the results in selected uniform variables

# How it works...

Compute shaders can be used only in a dedicated compute pipeline. They also cannot be executed (dispatched) inside render passes.

Compute shaders don't have any input nor output (user defined) variables passed from earlier or to later pipeline stages--it is the only stage in a compute pipeline. Uniform variables must be used for the source of compute shader data. Similarly, results of calculations performed in the compute shader can be stored only in the uniform variables.

There are some built-in input variables that provide information about the index of a given shader invocation within a local workgroup (through the `uvec3 gl_LocalInvocationID` variable), the number of workgroups dispatched at the same time (through the `uvec3 gl_NumWorkGroups` variable), or a number of the current workgroup (`uvec3 gl_WorkGroupID` variable). There is also a variable that uniquely identifies the current shader within all invocations in all workgroups--`uvec3 gl_GlobalInvocationID`. Its value is calculated like this:

[PRE11]

The size of the local workgroup is defined through the input layout qualifier. Inside the shader, the defined size is also available through the `uvec3 gl_WorkGroupSize` built-in variable.

In the following code, you can find a compute shader example that uses the `gl_GlobalInvocationID` variable to generate a simple, static fractal image:

[PRE12]

The preceding compute shader generates the following result when dispatched:

![](img/image_07_005.png)

# See also

*   The following recipe in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Creating a compute pipeline*

# Writing a vertex shader that multiplies vertex position by a projection matrix

Transforming geometry from local to clip space is usually performed by the vertex shader, though any other vertex processing stage (tessellation or geometry) may accomplish this task. The transformation is done by specifying model, view, and projection matrices and providing them from the application to the shaders as three separate matrices, or as one, joined model-view-projection matrix (in short MVP). The most common and easy way is to use a uniform buffer through which we can provide such a matrix.

# How to do it...

1.  Create a vertex shader in a text file called `shader.vert` (refer to the *Writing vertex shaders* recipe).
2.  Define an input variable (attribute) through which vertex positions will be provided to the vertex shader:

[PRE13]

3.  Define a uniform buffer with a variable of type `mat4` through which data for the combined model-view-projection matrix will be provided:

[PRE14]

4.  Inside a `void main()` function, calculate vertex position in the clip space by multiplying the `ModelViewProjectionMatrix` uniform variable by the `app_position` input variable and storing the result in the `gl_Position` built-in variable like this:

[PRE15]

# How it works...

When we prepare a geometry that will be drawn in a 3D application, the geometry is usually modeled in the local coordinate system--the one in which it is more convenient for the artist to create the model. However, the graphics pipeline expects the vertices to be transformed to a clip space, as it is easier (and faster) to perform many operations in this coordinate system. Usually it is the vertex shader that performs this transformation. For this, we need to prepare a matrix that represents a perspective or orthogonal projection. Transformation from the local space to the clip space is performed by just multiplying the matrix by the position of a vertex.

The same matrix, apart from the projection, may also contain other operations, commonly referred to as model-view transformations. And because drawn geometry may contain hundreds or thousands of vertices, it is usually more optimal to multiply model, view, and projection matrices in the application, and provide a single, concatenated MVP matrix to the shader which needs to perform only a single multiplication:

[PRE16]

The preceding shader requires the application to prepare a buffer in which data for the matrix is stored (refer to the *Creating a uniform buffer* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*). This buffer is then provided (in the current example) at a `0`^(th) binding to a descriptor set, which is later bound to the command buffer as the `0`^(th) set (refer to the *Updating descriptor sets* and *Binding descriptor sets* recipes from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*).

![](img/image_07_006.png)

# See also

*   In Chapter 5, *Descriptor Sets*, see the following recipes:
    *   *Creating a uniform buffer*
    *   *Updating descriptor sets*
    *   *Binding descriptor sets*
*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing vertex shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Creating a graphics pipeline*

# Using push constants in shaders

When we provide data to shaders, we usually use uniform buffers, storage buffers, or other types of descriptor resources. Unfortunately, updating such resources may not be too convenient, especially when we need to provide data that changes frequently.

For this purpose, push constants were introduced. Through them we can provide data in a simplified and much faster way than by updating descriptor resources. However, we need to fit into a much smaller amount of available space.

Accessing push constants in GLSL shaders is similar to using uniform buffers.

# How to do it...

1.  Create a shader file.
2.  Define a uniform block:
    1.  Provide a `push_constant `layout qualifier:
        `layout( push_constant )`
    2.  Use a `uniform` storage qualifier

4.  3.  Provide a unique name of the block
    4.  Inside the braces, define a set of uniform variables
    5.  Specify the name of the block instance `<instance name>`.
5.  Inside the `void main()` function, access uniform variables using a block instance name:

[PRE17]

# How it works...

Push constants are defined and accessed in a way similar to uniform blocks are specified in GLSL shaders, but there are some differences we need to remember:

1.  We need to use a `layout( push_constant )` qualifier before the definition of the block
2.  We must specify an instance name for the block
3.  We can define only one such block per shader
4.  We access push constant variables by preceding their name with the instance name of the block:

[PRE18]

Push constants are useful for providing small amounts of data that change frequently, such as the transformation matrix or current time value--updating the push constants block should be much faster than updating descriptor resources such as uniform buffers. We just need to remember about the data size which is much smaller than it is for descriptor resources. Specification requires push constants to store at least 128 bytes of data. Each hardware platform may allow for more storage, but it may not be considerably bigger.

Push constants can store at least 128 bytes of data.

An example of defining and using push constants in a fragment shader through which a color is provided may look like this:

[PRE19]

# See also

*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing a vertex shader that multiplies a vertex position by a projection matrix*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Creating a pipeline layout*
*   In [Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, see the following recipe:
    *   *Providing data to shaders through push constants*

# Writing texturing vertex and fragment shaders

Texturing is a common technique that significantly improves the quality of rendered images. It allows us to load an image and wrap it around the object like a wallpaper. It increases the memory usage, but saves the performance which would be wasted on processing much more complex geometry.

# How to do it...

1.  Create a vertex shader in a text file called `shader.vert` (refer to the *Writing vertex shaders* recipe).
2.  Apart from the vertex position, define an additional input variable (attribute) in the vertex shader through which texture coordinates are provided from the application:

[PRE20]

3.  In the vertex shader, define an output (varying) variable through which texture coordinates will be passed from the vertex shader to a fragment shader:

[PRE21]

4.  In the vertex shader's `void main()` function, assign the `app_tex_coordinates` variable to the `vert_tex_coordinates` variable:

[PRE22]

5.  Create a fragment shader (refer to the *Writing fragment shaders* recipe).
6.  In the fragment shader, define an input variable in which texture coordinates provided from the vertex shader will be passed:

[PRE23]

7.  Create a uniform `sampler2D` variable that will represent the texture which should be applied to the geometry:

[PRE24]

8.  Define an output variable in which the fragment's final color (read from the texture) will be stored:

[PRE25]

9.  In the fragment shader's `void main()` function, sample the texture and store the result in the `frag_color` variable:

[PRE26]

# How it works...

To draw an object, we need all its vertices. To be able to use a texture and apply it to the model, apart from vertex positions, we also need texture coordinates specified for each vertex. These attributes (position and texture coordinate) are passed to the vertex shader. It takes the position and transforms it to the clip space (if necessary), and passes the texture coordinates to the fragment shader:

[PRE27]

The texturing operation is performed in the fragment shader. The texture coordinates from all the vertices forming a polygon are interpolated and provided to the fragment shader. It uses these coordinates to read (sample) a color from the texture. This color is stored in the output and (potentially) in the attachment:

[PRE28]

Apart from providing texture coordinates to shaders, the application also needs to prepare the texture itself. Usually, this is done by creating a combined image sampler (refer to the *Creating a combined image sampler* recipe from [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*) and providing it to a descriptor set at `0`^(th) binding (in this sample). The Descriptor set must be bound to a `0`^(th) set index.

![](img/image_07_007.png)

# See also

*   In [Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, see the following recipes:
    *   *Creating a combined image sampler*
    *   *Updating descriptor sets*
    *   *Binding descriptor sets*
*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing vertex shaders*
    *   *Writing fragment shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Creating a graphics pipeline*

# Displaying polygon normals with a geometry shader

When rendering a geometry, we usually provide multiple attributes for each vertex--positions to draw the model, texture coordinates for texturing, and normal vectors for lighting calculation. Checking if all this data is correct may not be easy, but sometimes, when our rendering technique doesn't work as expected, it may be necessary.

In graphics programming, there are some debugging methods that are commonly used. Texture coordinates, which are usually two-dimensional, are displayed instead of the usual color. We can do the same with the normal vectors, but as they are three-dimensional, we can also display them in a form of lines. For this purpose, a geometry shader may be used.

# How to do it...

1.  Create a vertex shader called `normals.vert` (refer to the *Writing vertex shaders* recipe).
2.  Define an input variable in which the vertex position will be provided to the vertex shader:

[PRE29]

3.  Define a second input variable in which the vertex normal vector will be provided:

[PRE30]

4.  Define a uniform block with two matrices--one for a model-view transformation, the other for a projection matrix:

[PRE31]

5.  Define an output variable through which we will provide to a geometry shader a normal vector converted from a local space to a view space:

[PRE32]

6.  Convert a vertex position to a view space by multiplying the `ModelViewMatrix` variable and storing the result in the `gl_Position` built-in variable:

[PRE33]

7.  In a similar way, convert the vertex normal to a view space, scale the result by a chosen value, and store the result in the `vert_normal` output variable:

[PRE34]

8.  Create a geometry shader called `normal.geom` (refer to the *Writing geometry shaders* recipe).
9.  Define a `triangle` input primitive type:

[PRE35]

10.  Define an input variable through which the view-space vertex normal will be provided from the vertex shader:

[PRE36]

11.  Define a uniform block with two matrices--one for a model-view transformation, the other for a projection matrix:

[PRE37]

12.  Through an output layout qualifier, specify a `line_strip` as a generated primitive type with up to six vertices.

[PRE38]

13.  Define an output variable through which a color will be provided from the geometry shader to a fragment shader:

[PRE39]

14.  Inside the `void main()` function, use a variable of type `int` named `vertex` to loop over all the input vertices. Perform the following operations for each input vertex:
    1.  Multiply `ProjectionMatrix` by an input vertex position and store the result in the `gl_Position` built-in variable:
        `gl_Position = ProjectionMatrix * gl_in[vertex].gl_Position; `
    2.  In the `geom_color` output variable, store the desired color for the vertex normal at the contact point between geometry (vertex) and the vertex normal line:
        `geom_color = vec4( <chosen color> );`
    3.  Generate a new vertex by calling the `EmitVertex()` function.
    4.  Multiply `ProjectionMatrix` by the input vertex position offset by the `vert_normal` input variable. Store the result in the `gl_Position` built-in variable:
        `gl_Position = ProjectionMatrix * (gl_in[vertex].gl_Position + vert_normal[vertex]);`
    5.  Store the color of the vertex normal's end point in the `geom_color` output variable:
        `geom_color = vec4( <chosen color> );`
    6.  Generate a new vertex by calling the `EmitVertex()` function.
    7.  Generate a primitive (a line with two points) by calling the `EndPrimitive()` function.
15.  Create a fragment shader named `normals.frag` (refer to the *Writing fragment shaders* recipe).
16.  Define an input variable through which a color, interpolated between two vertices of a line generated by the geometry shader, will be provided to the fragment shader:

[PRE40]

17.  Define an output variable for the fragment's color:

[PRE41]

18.  Inside the `void main()` function, store the value of the `geom_color` input variable in the `frag_color` output variable:

[PRE42]

# How it works...

Displaying vertex normal vectors from the application side is performed in two steps: first we draw geometry in a normal way with the usual set of shaders. The second step is to draw the same model but with a pipeline object that uses the vertex, geometry, and fragment shaders specified in this recipe.

The vertex shader just needs to pass a vertex position and a normal vector to the geometry shader. It may transform both to the view-space, but the same operation can be performed in the geometry shader. The sample source code of the vertex shader that does the transformation provided through a uniform buffer is presented in the following code:

[PRE43]

In the preceding code, the position and normal vector are both transformed to the view space with a model-view matrix. If we intend to scale a model non-uniformly (not the same scale for all dimensions), the normal vector must be transformed using an inverse transpose of the model-view matrix.

The most important part of the code is performed inside geometry. It takes vertices that form the original primitive type (usually triangles), but outputs vertices forming line segments. It takes one input vertex, transforms it to a clip space and passes it further. The same vertex is used a second time, but this time it is offset by the vertex normal. After the translation, it is transformed to the clip space and passed to the output. These operations are performed for all vertices forming the original primitive. The source code for the whole geometry shader may look like this:

[PRE44]

The geometry shader takes vertices converted by the vertex shader to the view space and transforms them further to the clip space. This is done with a projection matrix provided through the same uniform buffer as the one used in a vertex shader. Why do we define two matrix variables in a single uniform buffer, if we use just one of them in the vertex shader and a second one in the geometry shader? Such an approach is more convenient, because we just need to create a single buffer and we need to bind only one descriptor set to the command buffer. In general, the less operations we perform or record in the command buffer, the more performance we achieve. So this approach should also be faster.

The fragment shader is simple as it only passes interpolated colors stored by the geometry shader:

[PRE45]

The result of using the preceding shaders to draw a geometry, along with a model drawn in a normal way, can be seen in the following image:

![](img/image_07_008.png)

# See also

*   The following recipes in this chapter:
    *   *Converting GLSL shaders to SPIR-V assemblies*
    *   *Writing vertex shaders*
    *   *Writing geometry shaders*
    *   *Writing fragment shaders*
*   In [Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, see the following recipes:
    *   *Creating a shader module*
    *   *Creating a graphics pipeline*