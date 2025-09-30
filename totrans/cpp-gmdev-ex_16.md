# Drawing Vulkan Objects

In the previous chapter, we created all the resources that were required for an object to be drawn. In this chapter, we will create the `ObjectRenderer` class, which will draw an object on the viewport. This class is used so that we have an actual geometric object to draw and view alongside our awesome purple viewport.

We will also learn how to synchronize CPU and GPU operations at the end of the chapter, which will remove the validation error that we got in [Chapter 11](ebc6fd68-325b-439e-9e11-8e01f818dd9b.xhtml), *Creating Object Resources*.

Before we set the scene up for rendering, we have to prepare one last thing for the geometry render; that is, the graphics pipeline. We will begin setting this up next.

In this chapter, we will cover the following topics:

*   Preparing the `GraphicsPipeline` class
*   The `ObjectRenderer` class
*   Changes to the `VulkanContext` class
*   The `Camera` class
*   Drawing an object
*   Synchronizing an object

# Preparing the GraphicsPipeline class

The graphics pipeline defines the pipeline an object should follow when it is drawn. As we discovered in [Chapter 2](ee788533-687d-4231-91a4-cb1de9ca01dd.xhtml), *Mathematics and Graphics Concepts*, there is a series of steps that we need to follow to draw an object:

![](img/14e58b29-0939-4abf-a065-4424953587e6.png)

In OpenGL, pipeline states can be changed at any time, just like we enabled and disabled blending when drawing text in [Chapter 8](ed5ec7d6-9257-48c4-9f66-3a2aca68eeeb.xhtml), *Enhancing Your Game with Collision, Loop, and Lighting*. However, changing states takes up a lot of system resources, which is why Vulkan discourages you from changing states at will. Therefore, you will have to set the pipeline states in advance for each object. Before you create a pipeline's state, you also need to create a pipeline layout that takes the descriptor set layout we created in the previous chapter. So, we will create the pipeline layout first.

Then, we also need to provide the shader SPIR-V files, which will have to be read to understand how to create the shader modules. So, add the functionality to the class. We then populate the graphics pipeline info, which will use the different shader modules that we created. We also specify the vertex input state, which will have information regarding the buffer's bindings and attributes, which we created earlier when defining the vertex struct.

The input assembly state also needs to be specified, which describes the kind of geometry to be drawn with the vertices. Note that we can draw points, lines, or triangles with the given set of vertices.

Additionally, we need to specify the viewport state, which describes the region of the framebuffer that will be rendered, as we can display part of the framebuffer to the viewport if necessary. In our case, we will be displaying the whole region to the viewport. We specify the rasterization state, which will perform depth testing and back-face culling, and convert the geometry to rasterized lines – which will be colored, as specified in the fragment shader.

The multisampling state will specify whether you want to enable multisampling to enable anti-aliasing. The depth and stencil states specify whether the depth and stencil tests are enabled and are to be performed on the object. The color blending state specifies whether blending is enabled or not. Finally, the dynamic state enables us to change some pipeline states dynamically without creating the pipeline again. We won't be using dynamic states for our implementation. With all this set, we can create the graphics pipeline for the object.

Let's begin by creating a new class for the graphics pipeline. In the `GraphicsPipeline.h` file, add the following:

```cpp
#include <vulkan\vulkan.h> 
#include <vector> 

#include <fstream> 

class GraphicsPipeline 
{ 
public: 
   GraphicsPipeline(); 
   ~GraphicsPipeline(); 

   VkPipelineLayout pipelineLayout; 
   VkPipeline graphicsPipeline; 

   void createGraphicsPipelineLayoutAndPipeline(VkExtent2D 
     swapChainImageExtent, VkDescriptorSetLayout descriptorSetLayout, 
     VkRenderPass renderPass); 

   void destroy(); 

private: 

   std::vector<char> readfile(const std::string& filename); 
   VkShaderModule createShaderModule(const std::vector<char> & code); 

   void createGraphicsPipelineLayout(VkDescriptorSetLayout 
      descriptorSetLayout); 
   void createGraphicsPipeline(VkExtent2D swapChainImageExtent, 
      VkRenderPass renderPass); 

}; 
```

We include the usual headers and also `fstream`, because we will need it for reading the shader files. We then create the class itself. In the `public` section, we will add the constructor and destructor. We create objects for storing the `pipelineLayout` and `graphicsPipeline` of the `VkPipelineLayout` and `VkPipeline` types respectively.

We create a new function called `createGraphicsPipelineLayoutAndPipeline`, which takes `VkExtent2D`, `VkDesriptorSetLayout`, and `VkRenderPass`, as this is required for creating both the layout and the actual pipeline itself. The function will internally be calling `createGraphicsPipelineLayout` and `createGraphicsPipeline`, which will create the layout and the pipeline respectively. These functions are added to the `private` section.

In the `public` section, we also have a function called `destroy`, which will destroy all the created resources. In the `private` section, we also have two more functions. The first is the `readFile` function, which reads the SPIR-V file, and the second is `createShaderModule`, which will create the shader module from the read shader file. Let's now move on to the `GraphicsPipeline.cpp` file:

```cpp
#include "GraphicsPipeline.h" 

#include "VulkanContext.h" 
#include "Mesh.h" 

GraphicsPipeline::GraphicsPipeline(){} 

GraphicsPipeline::~GraphicsPipeline(){} 
```

In the preceding code block, we include the `GraphicsPipeline.h`, `VulkanContext.h`, and `Mesh.h` files because they are required. We also add the implementation for the constructor and the destructor.

We then add the `createGraphicsPipelineLayoutAndPipeline` function, as follows:

```cpp
void GraphicsPipeline::createGraphicsPipelineLayoutAndPipeline(VkExtent2D 
  swapChainImageExtent, VkDescriptorSetLayout descriptorSetLayout, 
  VkRenderPass renderPass){ 

   createGraphicsPipelineLayout(descriptorSetLayout); 
   createGraphicsPipeline(swapChainImageExtent, renderPass); 

} 
```

The `createPipelineLayout` function is created as follows. We have to create a `createInfo` struct with the structure type, set the `descriptorLayout` and count, and then create the pipeline layout using the `vkCreatePipelineLayout` function:

```cpp
void GraphicsPipeline::createGraphicsPipelineLayout(VkDescriptorSetLayout 
  descriptorSetLayout){ 

   VkPipelineLayoutCreateInfo pipelineLayoutInfo = {}; 
   pipelineLayoutInfo.sType = VK_STRUCTURE_TYPEPIPELINE_                              LAYOUT_CREATE_INFO; 

// used for passing uniform objects and images to the shader 

pipelineLayoutInfo.setLayoutCount = 1; 
   pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; 

   if (vkCreatePipelineLayout(VulkanContext::getInstance()->
      getDevice()->logicalDevice, &pipelineLayoutInfo, nullptr, 
      &pipelineLayout) != VK_SUCCESS) { 

         throw std::runtime_error(" failed to create pieline 
            layout !"); 
   } 

} 
```

Before we add the create pipeline function, we will add the `readFile` and `createShaderModule` functions:

```cpp
std::vector<char> GraphicsPipeline::readfile(const std::string& filename) { 

   std::ifstream file(filename, std::ios::ate | std::ios::binary); 

   if (!file.is_open()) { 
         throw std::runtime_error(" failed to open shader file"); 
   } 
   size_t filesize = (size_t)file.tellg(); 

   std::vector<char> buffer(filesize); 

   file.seekg(0); 
   file.read(buffer.data(), filesize); 

   file.close(); 

   return buffer; 

}  
```

`readFile` takes a SPIR-V code file, opens and reads it, saves the contents of the file into a vector of char called `buffer`, and then returns it. We then add the `createShaderModule` function, as follows:

```cpp
VkShaderModule GraphicsPipeline::createShaderModule(const std::vector<char> & code) { 

   VkShaderModuleCreateInfo cInfo = {}; 

   cInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; 
   cInfo.codeSize = code.size(); 
   cInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); 

   VkShaderModule shaderModule; 
   if (vkCreateShaderModule(VulkanContext::getInstance()->getDevice()->
     logicalDevice, &cInfo, nullptr, &shaderModule) != VK_SUCCESS) { 
     throw std::runtime_error(" failed to create shader module !"); 
   } 

   return shaderModule; 
} 
```

To create the shader module, which is required for `ShaderStageCreateInfo` to create the pipeline, we need to populate the `ShaderModuleCreateInfo,` which takes the code and the size from the buffer to create the shader module. The shader module is created using the `vkCreateShaderModule` function, which takes the device and the `CreateInfo`. Once the shader module is created, it is returned. To create the pipeline, we have to create the following info structs: the shader stage info, the vertex input info, the input assembly struct, the viewport info struct, the rasterization info struct, the multisample state struct, the depth stencil struct (if required), the color blending struct, and the dynamic state struct.

So, let's create each, one after the other, starting with the shader stage struct. Add the `createGraphicsPipeline` function, and in it, we will create the pipeline:

```cpp
void GraphicsPipeline::createGraphicsPipeline(VkExtent2D swapChainImageExtent,  VkRenderPass renderPass) { 

... 

} 
```

In this function, we will now add the following, which will create the graphics pipeline.

# ShaderStageCreateInfo

To create the vertex shader, `ShaderStageCreateInfo`, we need to read the shader code first and create the shader module for it:

```cpp
   auto vertexShaderCode = readfile("shaders/SPIRV/basic.vert.spv"); 

   VkShaderModule vertexShadeModule = createShaderModule(vertexShaderCode); 
```

To read the shader file, we pass in the location of the shader file. Then, we pass the read code into the `createShaderModule` function, which will give us `vertexShaderModule`. We create the shader stage info struct for the vertex shader and pass in the stage, the shader module, and the name of the function to be used in the shader, which is `main`, in our case:

```cpp
   VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo = {}; 
   vertShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER
                                     _STAGE_CREATE_INFO; 
   vertShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; 
   vertShaderStageCreateInfo.module = vertexShadeModule; 
   vertShaderStageCreateInfo.pName = "main";  
```

Similarly, we will create the `ShaderStageCreateInfo` struct for the fragment shader:

```cpp
   auto fragmentShaderCode = readfile("shaders/SPIRV/basic.frag.spv"); 
   VkShaderModule fragShaderModule = createShaderModule
                                     (fragmentShaderCode); 

   VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo = {}; 

   fragShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINESHADER_                                      STAGE_CREATE_INFO; 
   fragShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT; 
   fragShaderStageCreateInfo.module = fragShaderModule; 
   fragShaderStageCreateInfo.pName = "main";
```

Note that the shader stage is set to `VK_SHADER_STAGE_FRAGMENT_BIT` to show that this is the fragment shader, and we also pass in `basic.frag.spv` as the file to read, which is the fragment shader file. We then create an array of `shaderStageCreateInfo` and add the two shaders to it for convenience:

```cpp
VkPipelineShaderStageCreateInfo shaderStages[] = {    
    vertShaderStageCreateInfo, fragShaderStageCreateInfo }; 
```

# VertexInputStateCreateInfo

In this info, we specify the input buffer binding and the attribute description:

```cpp
auto bindingDescription = Vertex::getBindingDescription(); 
auto attribiteDescriptions = Vertex::getAttributeDescriptions(); 

VkPipelineVertexInputStateCreateInfo vertexInputInfo = {}; 
vertexInputInfo.sType = VK_STRUCTURE_TYPE*PIPELINE
*                           VERTEX_INPUT_STATE_CREATE_INFO;
// initially was 0 as vertex data was hardcoded in the shader 
vertexInputInfo.vertexBindingDescriptionCount = 1; 
vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; 

vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t> 
                                                  (attribiteDescriptions.size());    
vertexInputInfo.pVertexAttributeDescriptions = attribiteDescriptions
                                               .data();
```

This is specified in the `Mesh.h` file under the vertex struct.

# InputAssemblyStateCreateInfo

Here, we specify the geometry we want to create, which is a triangle list. Add it as follows:

```cpp
   VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {}; 
   inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE*INPUT
*                             ASSEMBLY_STATE_CREATE_INFO; 
   inputAssemblyInfo.topology =  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; 
   inputAssemblyInfo.primitiveRestartEnable = VK_FALSE; 
```

# RasterizationStateCreateInfo

In this struct, we specify that depth clamping is enabled, which, instead of discarding the fragments if they are beyond the near and far planes, still keeps the value of that fragment and sets it equal to the near or far plane, even if that pixel is beyond either of these planes.

Discard the pixel in the rasterization stage by setting the value of `rasterizerDiscardEnable` to true or false. Set the polygon mode to either `VK_POLYGON_MODE_FILL` or `VK_POLYGON_MODE_LINE`. If it is set to line, then only a wireframe will be drawn; otherwise, the insides are also rasterized.

We can set the line width with the `lineWidth` parameter. Additionally, we can enable or disable back-face culling and then set the order of the front face winding by setting the `cullMode` and `frontFace` parameters.

We can alter the depth value by enabling it and adding a constant value to the depth, clamping it, or adding a slope factor. Depth biases are used in shadow maps, which we won't be using, so we won't enable depth bias. Add the struct and populate it as follows:

```cpp
   VkPipelineRasterizationStateCreateInfo rastStateCreateInfo = {}; 
   rastStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE*RASTERIZATION
*                               STATE_CREATE_INFO; 
   rastStateCreateInfo.depthClampEnable = VK_FALSE; 
   rastStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;  
   rastStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL; 
   rastStateCreateInfo.lineWidth = 1.0f; 
   rastStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT; 
   rastStateCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE; 
   rastStateCreateInfo.depthBiasEnable = VK_FALSE; 
   rastStateCreateInfo.depthBiasConstantFactor = 0.0f; 
   rastStateCreateInfo.depthBiasClamp = 0.0f; 
   rastStateCreateInfo.depthBiasSlopeFactor = 0.0f; 

```

# MultisampleStateCreateInfo

For our project, we won't be enabling multisampling for anti-aliasing. However, we will still need to create the struct:

```cpp
   VkPipelineMultisampleStateCreateInfo msStateInfo = {}; 
   msStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE*MULTISAMPLE
*                       STATE_CREATE_INFO; 
   msStateInfo.sampleShadingEnable = VK_FALSE; 
   msStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
```

We disable it by setting `sampleShadingEnable` to false and setting the sample count to `1`.

# Depth and stencil create info

Since we don't have a depth or stencil buffer, we don't need to create it. But when you have a depth buffer, you will need to add it to use the depth texture.

# ColorBlendStateCreateInfo

We set the color blending to false because it is not required for our project. To populate it, we have to first create the `ColorBlend` attachment state, which contains the configuration of each `ColorBlend` in each attachment. Then, we create `ColorBlendStateInfo`, which contains the overall blend state.

Create the `ColorBlendAttachment` state as follows. In this, we still specify the color write mask, which is the red, green, blue, and alpha bits, and set the attachment state to false, which disables blending for the framebuffer attachment:

```cpp
   VkPipelineColorBlendAttachmentState  cbAttach = {}; 
   cbAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | 
                             VK_COLOR_COMPONENT_G_BIT | 
                             VK_COLOR_COMPONENT_B_BIT | 
                             VK_COLOR_COMPONENT_A_BIT; 
   cbAttach.blendEnable = VK_FALSE; 
```

We create the actual blend struct, which takes the blend attachment info created, and we set the attachment count to `1` because we have a single attachment:

```cpp
   cbCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLORBLEND_                        STATE_CREATE_INFO; 
   cbCreateInfo.attachmentCount = 1; 
   cbCreateInfo.pAttachments = &cbAttach;  
```

# Dynamic state info

Since we don't have any dynamic states, this is not created.

# ViewportStateCreateInfo

In `ViewportStateCreateInfo`, we can specify the region of the framebuffer in which the output will be rendered to the viewport. So, we can render the scene but then only show some of it to the viewport. We can also specify a scissor rectangle, which will discard the pixels being rendered to the viewport.

However, we won't be doing anything fancy like that because we will render the whole scene to the viewport as it is. To define the viewport size and scissor size, we have to create the respective structs, as follows:

```cpp
   VkViewport viewport = {}; 
   viewport.x = 0; 
   viewport.y = 0; 
   viewport.width = (float)swapChainImageExtent.width; 
   viewport.height = (float)swapChainImageExtent.height; 
   viewport.minDepth = 0.0f; 
   viewport.maxDepth = 1.0f; 

   VkRect2D scissor = {}; 
   scissor.offset = { 0,0 }; 
   scissor.extent = swapChainImageExtent; 
```

For the viewport size and extent, we set them to the size of the `swapChain` image size in terms of width and height, starting from `(0, 0)`. We also set the minimum and maximum depth, which is normally between `0` and `1`.

For the scissor, since we want to show the whole viewport, we set the offset to `(0, 0)`, which indicates that we don't want the offset to start from where the viewport starts. Accordingly, we set the `scissor.extent` to the size of the `swapChain` image.

Now we can create `ViewportStateCreateInfo` function, as follows:

```cpp
   VkPipelineViewportStateCreateInfo vpStateInfo = {}; 
   vpStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE*VIEWPORT
*                       STATE_CREATE_INFO; 
   vpStateInfo.viewportCount = 1; 
   vpStateInfo.pViewports = &viewport; 
   vpStateInfo.scissorCount = 1; 
   vpStateInfo.pScissors = &scissor;  
```

# GraphicsPipelineCreateInfo

To create the graphics pipeline, we have to create the final `Info` struct, which we will populate with the `Info` structs we have created so far. So, add the struct as follows:

```cpp
   VkGraphicsPipelineCreateInfo gpInfo = {}; 
   gpInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; 

   gpInfo.stageCount = 2; 
   gpInfo.pStages = shaderStages; 

   gpInfo.pVertexInputState = &vertexInputInfo; 
   gpInfo.pInputAssemblyState = &inputAssemblyInfo; 
   gpInfo.pRasterizationState = &rastStateCreateInfo; 
   gpInfo.pMultisampleState = &msStateInfo; 
   gpInfo.pDepthStencilState = nullptr; 
   gpInfo.pColorBlendState = &cbCreateInfo; 
   gpInfo.pDynamicState = nullptr; 

gpInfo.pViewportState = &vpStateInfo; 
```

We also need to pass in the pipeline layout, render the pass, and specify whether there are any subpasses:

```cpp
   gpInfo.layout = pipelineLayout; 
   gpInfo.renderPass = renderPass; 
   gpInfo.subpass = 0; 
```

Now we can create the pipeline, as follows:

```cpp
  if (vkCreateGraphicsPipelines(VulkanContext::getInstance()->
    getDevice()->logicalDevice, VK_NULL_HANDLE, 1, &gpInfo, nullptr, 
    &graphicsPipeline) != VK_SUCCESS) { 
         throw std::runtime_error("failed to create graphics pipeline !!"); 
   } 
```

Additionally, make sure that you destroy the shader modules, as they are no longer required:

```cpp
   vkDestroyShaderModule(VulkanContext::getInstance()->
      getDevice()->logicalDevice, vertexShadeModule, nullptr); 
   vkDestroyShaderModule(VulkanContext::getInstance()->
      getDevice()->logicalDevice, fragShaderModule, nullptr); 
```

And that is all for the `createGraphicsPipeline` function. Finally, add the `destroy` function, which will destroy the pipeline and the layout:

```cpp
 void GraphicsPipeline::destroy(){ 

   vkDestroyPipeline(VulkanContext::getInstance()->
      getDevice()->logicalDevice, graphicsPipeline, nullptr); 
   vkDestroyPipelineLayout(VulkanContext::getInstance()->
      getDevice()->logicalDevice, pipelineLayout, nullptr); 

} 
```

# The ObjectRenderer class

With all the necessary classes created, we can finally create our `ObjectRenderer` class, which will render the mesh object to the scene.

Let's create a new class called `ObjectRenderer`. In `ObjectRenderer.h`, add the following:

```cpp
#include "GraphicsPipeline.h" 
#include "ObjectBuffers.h" 
#include "Descriptor.h" 

#include "Camera.h" 

class ObjectRenderer 
{ 
public: 
  void createObjectRenderer(MeshType modelType, glm::vec3 _position, 
     glm::vec3 _scale); 

   void updateUniformBuffer(Camera camera); 

   void draw(); 

   void destroy(); 

private: 

   GraphicsPipeline gPipeline; 
   ObjectBuffers objBuffers; 
   Descriptor descriptor; 

   glm::vec3 position; 
   glm::vec3 scale; 

};
```

We will include the descriptor, pipeline, and object buffer headers because they are required for the class. In the `public` section of the class, we will add objects of the three classes to define the pipeline, object buffers, and descriptors. We add four functions:

*   The first one is the `createObjectRenderer` function, which takes the model type, the position in which the object needs to be created, and the scale of the object.
*   Then, we have `updateUniformBuffer`, which will update the uniform buffer at every different frame and pass it to the shader. This takes the camera as a parameter because it is needed to get the view and perspective matrices. So, include the camera header as well.
*   We then have the `draw` function, which will be used to bind the pipeline, vertex, index, and descriptors to make the draw call.
*   We also have a `destroy` function to call the `destroy` functions of the pipeline, descriptors, and object buffers.

In the object's `Renderer.cpp` file, add the following `include` and the `createObjectRenderer` function:

```cpp
#include "ObjectRenderer.h" 
#include "VulkanContext.h" 
void ObjectRenderer::createObjectRenderer(MeshType modelType, glm::vec3 _position, glm::vec3 _scale){ 

uint32_t swapChainImageCount = VulkanContext::getInstance()->
                               getSwapChain()->swapChainImages.size(); 

VkExtent2D swapChainImageExtent = VulkanContext::getInstance()->
                                  getSwapChain()->swapChainImageExtent; 

   // Create Vertex, Index and Uniforms Buffer; 
   objBuffers.createVertexIndexUniformsBuffers(modelType); 

   // CreateDescriptorSetLayout 
     descriptor.createDescriptorLayoutSetPoolAndAllocate
     (swapChainImageCount); 
     descriptor.populateDescriptorSets(swapChainImageCount,
     objBuffers.uniformBuffers); 

   // CreateGraphicsPipeline 
   gPipeline.createGraphicsPipelineLayoutAndPipeline( 
       swapChainImageExtent, 
       descriptor.descriptorSetLayout, 
       VulkanContext::getInstance()->getRenderpass()->renderPass); 

   position = _position; 
   scale = _scale; 

 } 
```

We get the number of swap buffer images and their extents. Then, we create the vertex index and uniform buffers, we create and populate the descriptor set layout and sets, and then we create the graphics pipeline itself. Finally, we set the position and scale of the current object. Then, we add the `updateUniformBuffer` function. To get access to `SwapChain` and `RenderPass`, we will make some changes to the `VulkanContext` class:

```cpp
void ObjectRenderer::updateUniformBuffer(Camera camera){ 

   UniformBufferObject ubo = {}; 

   glm::mat4 scaleMatrix = glm::mat4(1.0f); 
   glm::mat4 rotMatrix = glm::mat4(1.0f); 
   glm::mat4 transMatrix = glm::mat4(1.0f); 

   scaleMatrix = glm::scale(glm::mat4(1.0f), scale); 
   transMatrix = glm::translate(glm::mat4(1.0f), position); 

   ubo.model = transMatrix * rotMatrix * scaleMatrix; 

   ubo.view = camera.viewMatrix 

   ubo.proj = camera.getprojectionMatrix 

   ubo.proj[1][1] *= -1; // invert Y, in Opengl it is inverted to
                            begin with 

   void* data; 
   vkMapMemory(VulkanContext::getInstance()->getDevice()->
     logicalDevice, objBuffers.uniformBuffersMemory, 0, 
     sizeof(ubo), 0, &data); 

   memcpy(data, &ubo, sizeof(ubo)); 

   vkUnmapMemory(VulkanContext::getInstance()->getDevice()->
     logicalDevice, objBuffers.uniformBuffersMemory); 

}
```

Here, we create a new `UniformBufferObject` struct called `ubo`. To do so, initialize the translation, rotation, and scale matrices. We then assign values for the scale and rotation matrices. After multiplying the scale, rotation, and translation matrices together, assign the result to the model matrix. From the `camera` class, we assign the view and projection matrices to `ubo.view` and `ubo.proj`. Then, we have to invert the *y* axis in the projection space because, in OpenGL, the *y* axis is already inverted. We now copy the updated `ubo` struct to the uniform buffer memory.

Next is the `draw` function:

```cpp
void ObjectRenderer::draw(){ 

   VkCommandBuffer cBuffer = VulkanContext::getInstance()->
                             getCurrentCommandBuffer(); 

   // Bind the pipeline 
   vkCmdBindPipeline(cBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
     gPipeline.graphicsPipeline); 

   // Bind vertex buffer to command buffer 
   VkBuffer vertexBuffers[] = { objBuffers.vertexBuffer }; 
   VkDeviceSize offsets[] = { 0 }; 

   vkCmdBindVertexBuffers(cBuffer, 
         0, // first binding index 
         1, // binding count 
         vertexBuffers, 
         offsets); 

   // Bind index buffer to the command buffer 
   vkCmdBindIndexBuffer(cBuffer, 
         objBuffers.indexBuffer, 
         0, 
         VK_INDEX_TYPE_UINT32); 

   //    Bind uniform buffer using descriptorSets 
   vkCmdBindDescriptorSets(cBuffer, 
         VK_PIPELINE_BIND_POINT_GRAPHICS, 
         gPipeline.pipelineLayout, 
         0, 
         1, 
         &descriptor.descriptorSet, 0, nullptr); 

   vkCmdDrawIndexed(cBuffer, 
         static_cast<uint32_t>(objBuffers.indices.size()), // no of indices 
         1, // instance count -- just the 1 
         0, // first index -- start at 0th index 
         0, // vertex offset -- any offsets to add 
         0);// first instance -- since no instancing, is set to 0  

} 
```

Before we actually make the draw call, we have to bind the graphics pipeline and pass in the vertex, index, and descriptor using the command buffer. To do this, we get the current command buffer and pass the commands through it. We will make changes to the `VulkanContext` class to get access to it as well.

We make the draw call using `vkCmdDrawIndexed`, in which we pass in the current command buffer, the index size, the instance count, the start of the index (which is `0`), the vertex offset (which is again `0`), and the location of the first index (which is `0`). Then, we add the `destroy` function, which basically just calls the `destroy` function of the pipeline, the descriptor, and the object buffer:

```cpp
 void ObjectRenderer::destroy() { 

   gPipeline.destroy(); 
   descriptor.destroy(); 
   objBuffers.destroy(); 

} 
```

# Changes to the VulkanContext class

To get access to `SwapChain`, `RenderPass`, and the current command buffer, we will add the following functions to the `VulkanContext.h` file under the `VulkanContext` class in the `public` section:

```cpp
   void drawBegin(); 
   void drawEnd(); 
   void cleanup(); 

   SwapChain* getSwapChain(); 
   Renderpass* getRenderpass(); 
   VkCommandBuffer getCurrentCommandBuffer();  
```

Then, in the `VulkanContext.cpp` file, add the implementation for accessing the values:

```cpp
SwapChain * VulkanContext::getSwapChain() { 

   return swapChain; 
} 

Renderpass * VulkanContext::getRenderpass() { 

   return renderPass; 
} 

VkCommandBuffer VulkanContext::getCurrentCommandBuffer() { 

   return curentCommandBuffer; 
} 
```

# The Camera class

We will create a basic camera class so that we can set the camera's position and set the view and projection matrices. This class will be very similar to the camera class created for the OpenGL project. The `camera.h` file is as follows:

```cpp
#pragma once 

#define GLM_FORCE_RADIAN 
#include <glm\glm.hpp> 
#include <glm\gtc\matrix_transform.hpp> 

class Camera 
{ 
public: 

   void init(float FOV, float width, float height, float nearplane, 
      float farPlane); 

void setCameraPosition(glm::vec3 position); 
   glm::mat4 getViewMatrix(); 
   glm::mat4 getprojectionMatrix(); 

private: 

   glm::mat4 projectionMatrix; 
   glm::mat4 viewMatrix; 
   glm::vec3 cameraPos; 

};
```

It has an `init` function, which takes the `FOV`, width, and height of the viewport, and the near and far planes to construct the projection matrix. We have a `setCameraPosition` function, which sets the location of the camera and two `getter` functions to get the camera view and projection matrices. In the `private` section, we have three local variables: two are for storing the projection and view matrices, and the third is a `vec3` for storing the camera's position.

The `Camera.cpp` file is as follows:

```cpp
#include "Camera.h" 

void Camera::init(float FOV, float width, float height, float nearplane, 
   float farPlane) { 

   cameraPos = glm::vec3(0.0f, 0.0f, 4.0f); 
   glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, 0.0f); 
   glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f); 

   viewMatrix = glm::mat4(1.0f); 
   projectionMatrix = glm::mat4(1.0f); 

   projectionMatrix = glm::perspective(FOV, width / height, nearplane, 
                      farPlane); 
   viewMatrix = glm::lookAt(cameraPos, cameraFront, cameraUp); 

} 

glm::mat4 Camera::getViewMatrix(){ 

   return viewMatrix; 

} 
glm::mat4 Camera::getprojectionMatrix(){ 

   return projectionMatrix; 
} 

void Camera::setCameraPosition(glm::vec3 position){ 

   cameraPos = position; 
} 
```

In the `init` function, we set the view and projection matrices, and then we add two getter functions and the `setCameraPosition` function.

# Drawing the object

Now that we have completed the prerequisites, let's draw a triangle:

1.  In `source.cpp`, include `Camera.h` and `ObjectRenderer.h`:

```cpp
#define GLFW_INCLUDE_VULKAN 
#include<GLFW/glfw3.h> 

#include "VulkanContext.h" 

#include "Camera.h" 
#include "ObjectRenderer.h" 
```

2.  In the `main` function, after initializing `VulkanContext`, create a new camera and an object to render, as follows:

```cpp
int main() { 

   glfwInit(); 

   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); 
   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); 

   GLFWwindow* window = glfwCreateWindow(1280, 720, 
                        "HELLO VULKAN ", nullptr, nullptr); 

   VulkanContext::getInstance()->initVulkan(window); 

   Camera camera; 
   camera.init(45.0f, 1280.0f, 720.0f, 0.1f, 10000.0f); 
   camera.setCameraPosition(glm::vec3(0.0f, 0.0f, 4.0f)); 

   ObjectRenderer object; 
   object.createObjectRenderer(MeshType::kTriangle, 
         glm::vec3(0.0f, 0.0f, 0.0f), 
         glm::vec3(0.5f)); 
```

3.  In the `while` loop, update the object's buffer and call the `object.draw` function:

```cpp
  while (!glfwWindowShouldClose(window)) { 

         VulkanContext::getInstance()->drawBegin(); 

         object.updateUniformBuffer(camera); 
         object.draw(); 

         VulkanContext::getInstance()->drawEnd(); 

         glfwPollEvents(); 
   }               
```

4.  When the program is done, call the `object.destroy` function:

```cpp
   object.destroy(); 

   VulkanContext::getInstance()->cleanup(); 

   glfwDestroyWindow(window); 
   glfwTerminate(); 

   return 0; 
} 
```

5.  Run the application and see a glorious triangle, as follows:

![](img/ab5584e7-f339-4be2-a71b-7576efe4a732.png)

Woohoo! Finally, we have a triangle. Well, we are still not quite done yet. Remember the annoying validation layer error that we keep getting? Take a look:

![](img/b5fc391a-688c-40f4-bb92-47d328b241e8.png)

It is time to understand why we are getting this error and what it actually means. This leads us to our final topic of this book: synchronization.

# Synchronizing the object

The process of drawing is actually asynchronous, meaning that the GPU might have to wait until the CPU has finished its current job. For example, using the constant buffer, we send instructions to the GPU to update each frame of the model view projection matrix. Now, if the GPU doesn't wait for the CPU to get the uniform buffer for the current frame, then the object would not be rendered correctly.

To make sure that the GPU only executes when the CPU has done its work, we need to synchronize the CPU and GPU. This can be done using two types synchronization objects:

*   The first is fences. Fences are synchronization objects that synchronize CPU and GPU operations.

*   We have a second kind of synchronization object, called semaphores. Semaphore objects synchronize GPU queues. In the current scene of one triangle that we are rendering, the graphics queue submits all the graphics commands, and then the presentation queue takes the image and presents it to the viewport. Of course, even this needs to be synchronized; otherwise, we will see scenes that haven't been fully rendered.

There are also events and barriers, which are other types of synchronization objects used for synchronizing work within a command buffer or a sequence of command buffers.

Since we haven't used any synchronization objects, the Vulkan validation layer is throwing errors and telling us that when we acquire an image from the SwapChain, we need to either use a fence or a semaphore to synchronize it.

In `VulkanContext.h`, in the `private` section, we will add the synchronization objects to be created, as follows:

```cpp
   const int MAX_FRAMES_IN_FLIGHT = 2; 
   VkSemaphore imageAvailableSemaphore;  
   VkSemaphore renderFinishedSemaphore;  
   std::vector<VkFence> inFlightFences;  
```

We have created two semaphores: one semaphore to signal when an image is available for us to render into, and another to signal when the rendering of the image has finished. We also created two fences to synchronize the two frames. In `VulkanContext.cpp`, under the `initVulkan` function, create the `Synchronization` object after the `DrawCommandBuffer` object:

```cpp
   drawComBuffer = new DrawCommandBuffer(); 
   drawComBuffer->createCommandPoolAndBuffer(swapChain->
      swapChainImages.size()); 

   // Synchronization 

   VkSemaphoreCreateInfo semaphoreInfo = {}; 
   semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; 

   vkCreateSemaphore(device->logicalDevice, &semaphoreInfo, 
      nullptr, &imageAvailableSemaphore); 
   vkCreateSemaphore(device->logicalDevice, &semaphoreInfo, 
      nullptr, &renderFinishedSemaphore); 

   inFlightFences.resize(MAX_FRAMES_IN_FLIGHT); 

   VkFenceCreateInfo fenceCreateInfo = {}; 
   fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; 
   fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; 

   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { 

         if (vkCreateFence(device->logicalDevice, &fenceCreateInfo, 
           nullptr, &inFlightFences[i]) != VK_SUCCESS) { 

         throw std::runtime_error(" failed to create synchronization 
            objects per frame !!"); 
         } 
   } 
```

We created the semaphore first using the `semaphoreCreatInfo` struct. We just have to set the struct type; we can create it using the `vkCreateSemaphore` function and pass in the logical device and the info struct.

Next, we create our fences. We resize the vector with the number of frames in flight, which is `2`. Then, we create the `fenceCreateInfo` struct and set the type of the struct. We now also signal the fences so that they are ready to be rendered. Then, we create the fences using `vkCreateFence` and pass in the logical device and the create fence info using a `for` loop. In the `DrawBegin` function, when we acquire the image, we pass in the `imageAvailable` semaphore to the function so that the semaphore will be signaled when the image is available for us to render into:

```cpp
   vkAcquireNextImageKHR(device->logicalDevice, 
         swapChain->swapChain, 
         std::numeric_limits<uint64_t>::max(), 
         imageAvailableSemaphore, // is  signaled 
         VK_NULL_HANDLE, 
         &imageIndex); 
```

Once an image is available to render into, we wait for the fence to be signaled so that we can start writing our command buffers:

```cpp
vkWaitForFences(device->logicalDevice, 1, &inFlightFences[imageIndex], 
   VK_TRUE, std::numeric_limits<uint64_t>::max());
```

We wait for the fence by calling `vkWaitForFences` and pass in the logical device, the fence count (which is `1`), and the fence itself. Then, we pass `TRUE` to wait for all fences, and pass in a timeout. Once the fence is available, we set it to unsignaled by calling `vkResetFence`, and then pass in the logical device, the fence count, and the fence:

```cpp
  vkResetFences(device->logicalDevice, 1, &inFlightFences[imageIndex]);  
```

The reset of the `DrawBegin` function remains the same so that we can begin recording the command buffer. Now, in the `DrawEnd` function, when it is time to submit the command buffer, we set the pipeline stage for `imageAvailableSemaphore` to wait on and set `imageAvailableSemaphore` to wait. We will set `renderFinishedSemaphore` to be signaled.

The `submitInfo` struct is changed accordingly:

```cpp
   // submit command buffer 
   VkSubmitInfo submitInfo = {}; 
   submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
   submitInfo.commandBufferCount = 1; 
   submitInfo.pCommandBuffers = &currentCommandBuffer; 

   // Wait for the stage that writes to color attachment 
   VkPipelineStageFlags waitStages[] = { VK_PIPELINE*STAGE*COLOR_
                                       ATTACHMENT_OUTPUT_BIT }; 
   // Which stage of the pipeline to wait 
   submitInfo.pWaitDstStageMask = waitStages; 

   // Semaphore to wait on before submit command execution begins 
   submitInfo.waitSemaphoreCount = 1; 
   submitInfo.pWaitSemaphores = &imageAvailableSemaphore;   

   // Semaphore to be signaled when command buffers have completed 
   submitInfo.signalSemaphoreCount = 1; 
   submitInfo.pSignalSemaphores = &renderFinishedSemaphore; 
```

The stage to wait on is `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT` for `imageAvailableSemaphore` to go from unsignaled to signaled. This will be signaled when the color buffer is written to. We then set `renderFinishedSemaphore` to be signaled so that the image will be ready for presenting. Submit the command and pass in the fence to show that the submission has been done:

```cpp
vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, inFlightFences[imageIndex]);
```

Once the submission is done, we can present the image. In the `presentInfo` struct, we set `renderFinishedSemaphore` to wait to go from an unsignaled state to a signaled state. We do this because, when the semaphore is signaled, the image will be ready for presentation:

```cpp
   // Present frame 
   VkPresentInfoKHR presentInfo = {}; 
   presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR; 

   presentInfo.waitSemaphoreCount = 1; 
   presentInfo.pWaitSemaphores = &renderFinishedSemaphore;  

   presentInfo.swapchainCount = 1; 
   presentInfo.pSwapchains = &swapChain->swapChain; 
   presentInfo.pImageIndices = &imageIndex; 

   vkQueuePresentKHR(device->presentQueue, &presentInfo); 
   vkQueueWaitIdle(device->presentQueue);  

```

In the `cleanup` function in `VulkanContext`, make sure that you destroy the semaphores and fences, as follows:

```cpp
 void VulkanContext::cleanup() { 

   vkDeviceWaitIdle(device->logicalDevice); 

   vkDestroySemaphore(device->logicalDevice, 
      renderFinishedSemaphore, nullptr); 
   vkDestroySemaphore(device->logicalDevice, 
      imageAvailableSemaphore, nullptr); 

   // Fences and Semaphores 
   for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) { 

        vkDestroyFence(device->logicalDevice, inFlightFences[i], nullptr); 
   } 

   drawComBuffer->destroy(); 
   renderTarget->destroy(); 
   renderPass->destroy(); 
   swapChain->destroy(); 

   device->destroy(); 

   valLayersAndExt->destroy(vInstance->vkInstance, isValidationLayersEnabled); 

   vkDestroySurfaceKHR(vInstance->vkInstance, surface, nullptr);   
   vkDestroyInstance(vInstance->vkInstance, nullptr); 

} 
```

Now build and run the application in debug mode and see that the validation layer has stopped complaining:

![](img/fb29545e-ac23-43a4-968b-a5260e027511.png)

Now draw other objects as well, such as a quad, cube, and sphere, by changing the `source.cpp` file as follows:

```cpp
#define GLFW_INCLUDE_VULKAN
#include<GLFW/glfw3.h>
#include "VulkanContext.h"
#include "Camera.h"
#include "ObjectRenderer.h"
int main() {

glfwInit();

glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

GLFWwindow* window = glfwCreateWindow(1280, 720, "HELLO VULKAN ", 
                     nullptr, nullptr);
VulkanContext::getInstance()->initVulkan(window);

Camera camera;
camera.init(45.0f, 1280.0f, 720.0f, 0.1f, 10000.0f);
camera.setCameraPosition(glm::vec3(0.0f, 0.0f, 4.0f));
ObjectRenderer tri;
tri.createObjectRenderer(MeshType::kTriangle,
glm::vec3(-1.0f, 1.0f, 0.0f),
glm::vec3(0.5f));

ObjectRenderer quad;
quad.createObjectRenderer(MeshType::kQuad,
glm::vec3(1.0f, 1.0f, 0.0f),
glm::vec3(0.5f));

ObjectRenderer cube;
cube.createObjectRenderer(MeshType::kCube,
glm::vec3(-1.0f, -1.0f, 0.0f),
glm::vec3(0.5f));

ObjectRenderer sphere;
sphere.createObjectRenderer(MeshType::kSphere,
glm::vec3(1.0f, -1.0f, 0.0f),
glm::vec3(0.5f));

while (!glfwWindowShouldClose(window)) {

VulkanContext::getInstance()->drawBegin();

// updatetri.updateUniformBuffer(camera);
quad.updateUniformBuffer(camera);
cube.updateUniformBuffer(camera);
sphere.updateUniformBuffer(camera);

// draw command
tri.draw();
quad.draw();
cube.draw();
sphere.draw();
VulkanContext::getInstance()->drawEnd();
glfwPollEvents();
}
tri.destroy();
quad.destroy();
cube.destroy();
sphere.destroy();
VulkanContext::getInstance()->cleanup();
glfwDestroyWindow(window);
glfwTerminate();
return 0;
}
```

And this should be the final output, with all the objects rendered:

![](img/a298e7f5-a683-46ee-851e-1f49d7253608.png)

You can also add custom geometries that can be loaded from a file.

In addition to this, now that you have different shapes to render, you can add physics and try to replicate the physics game made in OpenGL, and port the game to use the Vulkan rendering API.

Furthermore, the code can be extended to include the depth buffer, adding textures to the object, and more.

# Summary

So, this is the final summary of this book. In this book, we journeyed from creating a basic game in Simple and Fast Multimedia Library (SFML), which uses OpenGL for rendering, to showing how a rendering API fits into the whole scheme when making a game.

We then created a complete physics-based game from the ground up, using our own mini game engine. Apart from just drawing objects using the high-level OpenGL graphics API, we also added bullet physics to take care of game physics and contact detection between game objects. We also added some text rendering to make the score visible to the player, and we also learned about basic lighting to do lighting calculations for our small scene in order to make the scene a little more interesting.

Finally, we moved on to the Vulkan rendering API, which is a low-level graphics library. In comparison to OpenGL, which we used to make a small game by the end of [Chapter 3](7ae25a2f-fcf6-4501-a5f3-e5b7fb6e27c3.xhtml), *Setting Up Your Game*, in Vulkan, at the end of four chapters, we were able to render a basic geometric objects. However, with Vulkan, we have complete access to the GPU, which gives us more freedom to tailor the engine based on the game's requirements.

If you have come this far, then congratulations! I hope you enjoyed going through this book and that you will continue to expand your knowledge of the SFML, OpenGL, and Vulkan projects.

Best wishes.

# Further Reading

To learn more, I wholeheartedly recommend the Vulkan tutorial website: [https://vulkan-tutorial.com/](https://vulkan-tutorial.com/). The tutorial also covers how to add textures, depth buffers, model loading, and mipmaps. The code for the book is based on this tutorial, so it should be easy to follow and should help you take the Vulkan code base in the book further:

![](img/e6b960d7-7c8f-4186-bfd6-c1d914324702.png)

The source code for the Doom 3 Vulkan renderer is available at [https://github.com/DustinHLand/vkDOOM3](https://github.com/DustinHLand/vkDOOM3)—it is fun to see the code in practice:

![](img/e7c5ce9e-a689-4545-a3db-5e9bfce2cf41.png)

I also recommend reading the blog at [https://www.fasterthan.life/blog](https://www.fasterthan.life/blog), as it goes through the journey of porting the Doom 3 OpenGL code to Vulkan. In this book, we let Vulkan allocate and deallocate resources. This blog goes into detail about how memory management is done in Vulkan.