# Creating Object Resources

In the previous chapter, we got our clear screen working and created the Vulkan instance. We also created the logical device, the swapchain, the render targets, and the views, as well as the draw command buffer, to record and submit commands to the GPU. Using it, we were able to have a purple clear screen. We haven't drawn any geometry yet, but we are now ready to do so.

In this chapter, we will get most of the things that we need ready to render the geometries. We have to create vertex, index, and uniform buffers. The vertex, index, and uniform buffers will have information regarding the vertex attributes, such as position, color, normal and texture coordinates; index information will have the indices of the vertices we want to draw, and uniform buffers will have information such as a novel view projection matrix.

We will need to create a descriptor set and layout, which will specify to which shader stage the uniform buffers are bound.

We also have to generate the shaders that will be used to draw the geometry.

To create both the object buffers and descriptor sets and layouts, we will create new classes so that they are compartmentalized and we can understand how they are related. Before we hop on the object buffer class, we will add the `Mesh` class that we created in the OpenGL project, and we will use the same class and add minor changes to it. The `Mesh` class has information regarding the vertex and index information for the different geometry shapes we want to draw.

We will cover the following topics in this chapter:

*   Updating the `Mesh` class for Vulkan
*   Creating the `ObjectBuffers` class
*   Creating the `Descriptor` class
*   Creating the SPIR-V shader binary

# Updating the Mesh class for Vulkan

In the `Mesh.h` file, we just have to add a few lines of code to specify `InputBindingDescription` and `InputAttributeDescription`. In `InputBindingDesciption`, we specify the binding location, the stride of the data itself, and the input rate, which specifies whether the data is per vertex or per instance. In the `Mesh.h` file in the OpenGL project, we will just add functions to the `Vertex` struct:

```cpp
 struct Vertex { 

   glm::vec3 pos; 
   glm::vec3 normal; 
   glm::vec3 color; 
glm::vec2 texCoords; 

}; 
```

So, in the `Vertex` struct, add the function to retrieve `AttributeDesciption`:

```cpp
   static VkVertexInputBindingDescription getBindingDescription() { 

         VkVertexInputBindingDescription bindingDescription = {}; 

         bindingDescription.binding = 0;  
         bindingDescription.stride = sizeof(Vertex); 
         bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; 

         return bindingDescription; 
} 

```

In the function, `VertexInputBindingDescriptor` specifies that the binding is at the 0^(th) index, the stride is equal to the size of the `Vertex` struct itself, and the input rate is `VK_VERTEX_INPUT_RATE_VERTEX`, which is per vertex. The function just returns the created binding description.

Since we have four attributes in the vertex struct, we have to create an attribute descriptor for each one. Add the following function to the `Vertex` struct as well, which returns an array of four input attribute descriptors. For each attribute descriptor, we have to specify the binding location, which is `0` as specified in the binding description, the layout location for each attribute, the format of the data type, and the offset from the start of the `Vertex` struct:

```cpp
static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() { 

   std::array<VkVertexInputAttributeDescription, 4> 
   attributeDescriptions = {}; 

   attributeDescriptions[0].binding = 0; // binding index, it is 0 as 
                                            specified above 
   attributeDescriptions[0].location = 0; // location layout

   // data format
   attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; 
   attributeDescriptions[0].offset = offsetof(Vertex, pos); // bytes             
      since the start of the per vertex data 

   attributeDescriptions[1].binding = 0; 
   attributeDescriptions[1].location = 1; 
   attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; 
   attributeDescriptions[1].offset = offsetof(Vertex, normal); 

   attributeDescriptions[2].binding = 0; 
   attributeDescriptions[2].location = 2; 
   attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT; 
   attributeDescriptions[2].offset = offsetof(Vertex, color); 

   attributeDescriptions[3].binding = 0; 
   attributeDescriptions[3].location = 3; 
   attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT; 
   attributeDescriptions[3].offset = offsetof(Vertex, texCoords); 

   return attributeDescriptions; 
}   
```

We will also create a new struct in the `Mesh.h` file to organize the uniform data information. So, create a new struct called `UniformBufferObject`:

```cpp
struct UniformBufferObject { 

   glm::mat4 model; 
   glm::mat4 view; 
   glm::mat4 proj; 

}; 
```

At the top of the `Mesh.h` file, we will also include two `define` statements to tell `GLM` to use radians instead of degrees, and to use the normalized depth value:

```cpp
#define GLM_FORCE_RADIAN 
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 
```

That is all for `Mesh.h`. The `Mesh.cpp` file doesn't get modified at all.

# Creating the ObjectBuffers class

To create object-related buffers, such as vertex, index, and uniform, we will create a new class called `ObjectBuffers`. In the `ObjectBuffers.h` file, we will add the required `include` statements:

```cpp
#include <vulkan\vulkan.h> 
#include <vector> 

#include "Mesh.h"  
```

Then, we will create the class itself. In the public section, we will add the constructor and the destructor and add the required data types for creating vertex, index, and uniform buffers. We add a vector of the data vertex to set the vertex information of the geometry, create a `VkBuffer` instance called `vertexBuffer` to store the vertex buffer, and create a `VkDeviceMemory` instance called `vertexBufferMemory`:

*   `VkBuffer`: This is the handle to the object buffer itself.
*   `VkDeviceMemory`: Vulkan operates on memory data in the device's memory through the `DeviceMemory` object.

Similarly, we create a vector to store indices, and create an `indexBuffer` and an `indexBufferMemory` object, just as we did for vertex.

For the uniform buffer, we only create `uniformBuffer` and `uniformBuffer` memory as a vector is not required.

We add a `createVertexIndexUniformBuffers` function that takes in a `Mesh` type, and the vertices and indices will be set based on it.

We also add a destroy function to destroy the Vulkan object we created.

In the private section, we add three functions, which `createVertexIndexUniformBuffers` will call to create the buffers. That is all for the `ObjectBuffers.h` file. So, the `ObjectBuffers` class should be like this:

```cpp
class ObjectBuffers 
{ 
public: 
   ObjectBuffers(); 
   ~ObjectBuffers(); 

   std::vector<Vertex> vertices; 
   VkBuffer vertexBuffer; 
   VkDeviceMemory vertexBufferMemory; 

   std::vector<uint32_t> indices; 
   VkBuffer indexBuffer; 
   VkDeviceMemory indexBufferMemory; 

   VkBuffer uniformBuffers; 
   VkDeviceMemory uniformBuffersMemory; 

   void createVertexIndexUniformsBuffers(MeshType modelType); 
   void destroy(); 

private: 

   void createVertexBuffer(); 
   void createIndexBuffer(); 
   void createUniformBuffers(); 

}; 

```

Next, let's move on to the `ObjectBuffers.cpp` file. In this file, we include the headers and create the constructor and destructor:

```cpp
#include "ObjectBuffers.h" 
#include "Tools.h" 
#include "VulkanContext.h" 

ObjectBuffers::ObjectBuffers(){} 

ObjectBuffers::~ObjectBuffers(){} 
```

`Tools.h` is included as we will be adding some more functionality to it that we will use. Next, we will create the `createVertexIndexUniformBuffers` function:

```cpp
void ObjectBuffers::createVertexIndexUniformsBuffers(MeshType modelType){ 

   switch (modelType) { 

         case kTriangle: Mesh::setTriData(vertices, indices); break; 
         case kQuad: Mesh::setQuadData(vertices, indices); break; 
         case kCube: Mesh::setCubeData(vertices, indices); break; 
         case kSphere: Mesh::setSphereData(vertices, indices); break; 

   } 

    createVertexBuffer(); 
    createIndexBuffer(); 
    createUniformBuffers(); 

}
```

Similar to the OpenGL project, we will add a `switch` statement to set the vertex and index data depending on the mesh type. Then we call the `createVertexBuffer`, 
`createIndexBuffer`, and `createUniformBuffers` functions to set the respective buffers. We will create the `createVertexBuffer` function first.

To create the vertex buffer, it is better if we create the buffer on the device that is on the **GPU** itself. Now, the **GPU** has two types of memories: **HOST VISIBLE** and **DEVICE LOCAL**. **HOST VISIBLE** is a part of the GPU memory that the CPU has access to. This memory is not very large, so it is used for storing up to 250 MB of data.

For larger chunks of data, such as vertex and index data, it is better to use the **DEVICE LOCAL** memory, which the CPU doesn't have access to.

So, how do you transfer data to the `DEVICE LOCAL` memory? Well, we first have to copy the data to the **HOST VISIBLE** section on the **GPU**, and then copy it to the **DEVICE LOCAL** memory. So, we first create what is called a staging buffer, copy the vertex data into it, and then copy the staging buffer to the actual vertex buffer:

![](img/cc52a322-ee49-4ab3-b470-b047522a29f1.png)

(Source: [https://www.youtube.com/watch?v=rXSdDE7NWmA](https://www.youtube.com/watch?v=rXSdDE7NWmA))

Let's add functionality into the `VkTool` file to create the different kinds of buffers. With this, we can create both the staging buffer and vertex buffer itself. So, in the `VkTools.h` file in the `vkTools` namespace, add a new function called `createBuffer`. This function takes in five parameters:

*   The first is `VkDeviceSize`, which is the size of the data for which the buffer is to be created.
*   The second is the `usage` flag, which tells us what the buffer is going to be used for.

*   The third is the memory properties where we want to create the buffer; this is where we will specify whether we want it in the HOST VISIBLE section or the DEVICE LOCAL area.
*   The fourth is the buffer itself.
*   The fifth is the buffer memory to bind the buffer to the following:

```cpp
namespace vkTools { 

   VkImageView createImageView(VkImage image, 
         VkFormat format, 
         VkImageAspectFlags aspectFlags); 

   void createBuffer(VkDeviceSize size, 
         VkBufferUsageFlags usage, 
         VkMemoryPropertyFlags properties, 
         VkBuffer &buffer, 
         VkDeviceMemory& bufferMemory); 
} 
```

In the `VKTools.cpp` file, we add the functionality for creating the buffer and binding it to `bufferMemory`. In the namespace, add the new function:

```cpp
   void createBuffer(VkDeviceSize size, 
         VkBufferUsageFlags usage, 
         VkMemoryPropertyFlags properties, 
         VkBuffer &buffer, // output 
         VkDeviceMemory& bufferMemory) { 

// code  
} 
```

Before binding the buffer, we create the buffer itself. Hence, we populate the `VkBufferCreateInfo` struct as follows:

```cpp
   VkBufferCreateInfo bufferInfo = {}; 
   bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; 
   bufferInfo.size = size; 
   bufferInfo.usage = usage; 
   bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; 

   if (vkCreateBuffer(VulkanContext::getInstance()->
      getDevice()->logicalDevice, &bufferInfo, 
      nullptr, &buffer) != VK_SUCCESS) { 

         throw std::runtime_error(" failed to create 
           vertex buffer "); 
   }
```

The struct takes the usual type first, and then we set the buffer size and usage. We also need to specify the buffer sharing mode, because a buffer can be shared between queues, such as graphics and compute, or could be exclusive to one queue. So, here we specify that the buffer is exclusive to the current queue.

Then, the buffer is created by calling `vkCreateBuffer` and passing in `logicalDevice` and `bufferInfo`. Next, to bind the buffer, we have to get the suitable memory type for our specific use of the buffer. So, first we have to get the memory requirements for the kind of buffer we are creating. The requisite memory requirement is received by calling the `vkGetBUfferMemoryRequirements` function, which takes in the logical device, the buffer, and the memory requirements get stored in a variable type called `VkMemoryRequirements`.

We get the memory requirements as follows:

```cpp
   VkMemoryRequirements memrequirements; 
   vkGetBufferMemoryRequirements(VulkanContext::getInstance()->getDevice()->
     logicalDevice, buffer, &memrequirements);  
```

To bind memory, we have to populate the `VkMemoryAllocateInfo` struct. It requires the allocation size and the memory index of the type of memory. Each GPU has a different memory type index, with a different heap index and memory type. These are the corresponding values for 1080Ti:

![](img/48567b6b-285e-4469-8643-0b026dd39363.png)

We will now add a new function in `VkTools` to get the correct kind of memory index for our buffer usage. So, add a new function in `VkTool.h` under the `vkTools` namespace, called `findMemoryTypeIndex`:

```cpp
uint32_t findMemoryTypeIndex(uint32_t typeFilter, VkMemoryPropertyFlags 
    properties); 
```

It takes two parameters, which are the memory type bits available and the memory properties that we need. Add the implementation for the `findMemoryTypeIndex` function to the `VkTools.cpp` file. Under the namespace, add the following function:

```cpp
uint32_t findMemoryTypeIndex(uint32_t typeFilter, VkMemoryPropertyFlags properties) { 

   //-- Properties has two arrays -- memory types and memory heaps 
   VkPhysicalDeviceMemoryProperties memProperties; 
     vkGetPhysicalDeviceMemoryProperties(VulkanContext::
     getInstance()->getDevice()->physicalDevice, 
     &memProperties); 

   for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) { 

         if ((typeFilter & (1 << i)) &&  
             (memProperties.memoryTypes[i].propertyFlags &                                 
              properties) == properties) { 

                     return i; 
               } 
         } 

         throw std::runtime_error("failed to find 
            suitable memory type!"); 
   } 
```

This function gets the device's memory properties using the `vkGetPhysicalDeviceMemoryProperties` function, and populates the memory properties of the physical device.

The memory properties get information regarding the memory heap and memory type for each index. From all the available indices, we choose what is required for our purposes and return the values. Once the function has been created, we can go back to binding the buffer. So, continuing with our `createBuffer` function, add the following to it in order to bind the buffer to the memory:

```cpp
   VkMemoryAllocateInfo allocInfo = {}; 
   allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; 
   allocInfo.allocationSize = memrequirements.size; 
   allocInfo.memoryTypeIndex = findMemoryTypeIndex(memrequirements.
                               memoryTypeBits, properties); 

   if (vkAllocateMemory(VulkanContext::getInstance()->
      getDevice()->logicalDevice, &allocInfo, nullptr, 
      &bufferMemory) != VK_SUCCESS) { 

         throw std::runtime_error("failed to allocate 
            vertex buffer memory"); 
   } 

   vkBindBufferMemory(VulkanContext::getInstance()->
      getDevice()->logicalDevice, buffer, 
      bufferMemory, 0); 
```

After all that, we can go back to `ObjectBuffers` to actually create the `createVertexBuffers` function. So, create the function as follows:

```cpp
void ObjectBuffers::createVertexBuffer() { 
// code 
} 

```

In it, we will create the staging buffer first, copy the vertex data into it, and then copy the staging buffer into the vertex buffer. In the function, we first get the total buffer size, which is the number of vertices and the size of the data stored per vertex:

```cpp
VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size(); 
```

Next, we create the staging buffer and `stagingBufferMemory` to bind the staging buffer to it:

```cpp
VkBuffer stagingBuffer; 
VkDeviceMemory stagingBufferMemory; 
```

And then we call the newly created `createBuffer` in `vkTools` to create the buffer:

```cpp
vkTools::createBuffer(bufferSize, 
   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,  
   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
   stagingBuffer, stagingBufferMemory);
```

In it, we pass in the size, usage, memory type we want, and the buffer and buffer memory. `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` indicates that the buffer is going to be used as part of a source transfer command when data is transferred.

`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` specifies that we want this to be allocated in the host-visible (CPU) memory space on the GPU.

`VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` means that CPU cache management is not done by us, but by the system. This will make sure the mapped memory matches the allocated memory. Next, we use `vkMapMemory` to get a host pointer to the staging buffer and create a void pointer called data. Then, call `vkMapMemory` to get the pointer to the mapped memory:

```cpp
   void* data; 

   vkMapMemory(VulkanContext::getInstance()->getDevice()->
      logicalDevice, stagingBufferMemory, 
         0, // offet 
         bufferSize,// size 
         0,// flag 
         &data);  
```

`VkMapMemory` takes the logical device, the staging buffer binding, we specify 0 for the offset, and pass the buffer size. There are no special flags, so we pass in `0` and get the pointer to the mapped memory. We use `memcpy` to copy the vertex data to the data pointer:

```cpp
memcpy(data, vertices.data(), (size_t)bufferSize);  
```

We unmap the staging memory once host access to it is no longer required:

```cpp
vkUnmapMemory(VulkanContext::getInstance()->getDevice()->logicalDevice, 
   stagingBufferMemory); 
```

Now that the data is stored in the staging buffer, let's next create the vertex buffer and bind it to `vertexBufferMemory`:

```cpp
// Create Vertex Buffer 
   vkTools::createBuffer(bufferSize, 
         VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,  
         vertexBuffer, 
         vertexBufferMemory);
```

We use the `createBuffer` function to create the vertex buffer. We pass in the buffer size. For the buffer usage, we specify that it is used as the destination of the transfer command when we transfer the staging buffer to it, and it will be used as the vertex buffer. For the memory property, we want this to be created in `DEVICE_LOCAL` for the best performance. Pass the vertex buffer and vertex buffer memory to bind the buffer to the memory. Now, we have to copy the staging buffer to the vertex buffer.

Copying buffers on the GPU has to be done using transfer queues and command buffers. We could get the transfer queue to do the transfer in the same way as we retrieved the graphics and presentation queues. The good news is that we don't need to, because all graphics and compute queues also support transfer functionality, so we will use the graphics queue for it.

We will create two helper functions in the `vkTools` namespace for creating and destroying temporary command buffers. So, in the `VkTools.h` file, add two functions in the namespace for the beginning and ending single-time commands:

```cpp
VkCommandBuffer beginSingleTimeCommands(VkCommandPool commandPool); 
   void endSingleTimeCommands(VkCommandBuffer commandBuffer, 
   VkCommandPool commandPool);  
```

Basically, `beginSingleTimeCommands` returns a command buffer for us to use, and `endSingleTimeCommands` destroys the command buffer. In the `VkTools.cpp` file, under the namespace, add these two functions:

```cpp
   VkCommandBuffer beginSingleTimeCommands(VkCommandPool commandPool) { 

         //-- Alloc Command buffer   
         VkCommandBufferAllocateInfo allocInfo = {}; 

         allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND*BUFFER
*                           ALLOCATE_INFO; 
         allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; 
         allocInfo.commandPool = commandPool; 
         allocInfo.commandBufferCount = 1; 

         VkCommandBuffer commandBuffer; 
         vkAllocateCommandBuffers(VulkanContext::getInstance()->
           getDevice()->logicalDevice, 
           &allocInfo, &commandBuffer); 

         //-- Record command buffer 

         VkCommandBufferBeginInfo beginInfo = {}; 
         beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; 
         beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; 

         //start recording 
         vkBeginCommandBuffer(commandBuffer, &beginInfo); 

         return commandBuffer; 

   } 

   void endSingleTimeCommands(VkCommandBuffer commandBuffer, 
      VkCommandPool commandPool) { 

         //-- End recording 
         vkEndCommandBuffer(commandBuffer); 

         //-- Execute the Command Buffer to complete the transfer 
         VkSubmitInfo submitInfo = {}; 
         submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
         submitInfo.commandBufferCount = 1; 
         submitInfo.pCommandBuffers = &commandBuffer; 

         vkQueueSubmit(VulkanContext::getInstance()->
            getDevice()->graphicsQueue, 1, &submitInfo, 
            VK_NULL_HANDLE); 

         vkQueueWaitIdle(VulkanContext::getInstance()->
            getDevice()->graphicsQueue); 

         vkFreeCommandBuffers(VulkanContext::getInstance()->
            getDevice()->logicalDevice, commandPool, 1, 
            &commandBuffer); 

   } 
```

We have already looked at how to create and destroy command buffers. If you have any questions, you can refer to [Chapter 11](58a38eaf-b67b-4425-b5d6-80efaf4970ad.xhtml), *Preparing the Clear Screen*. Next, in the `Vktools.h` file, we will add the functionality to copy a buffer. Add a new function under the namespace:

```cpp
   VkCommandBuffer beginSingleTimeCommands(VkCommandPool commandPool); 
   void endSingleTimeCommands(VkCommandBuffer commandBuffer, 
      VkCommandPool commandPool); 

   void copyBuffer(VkBuffer srcBuffer, 
         VkBuffer dstBuffer, 
         VkDeviceSize size);
```

The `copyBuffer` function takes a source buffer, a destination buffer, and the buffer size as input. Now, add this new function to the `VkTools.cpp` file:

```cpp
void copyBuffer(VkBuffer srcBuffer, 
         VkBuffer dstBuffer, 
         VkDeviceSize size) { 

QueueFamilyIndices qFamilyIndices = VulkanContext::getInstance()->
   getDevice()->getQueueFamiliesIndicesOfCurrentDevice(); 

   // Create Command Pool 
   VkCommandPool commandPool; 

   VkCommandPoolCreateInfo cpInfo = {}; 

   cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; 
   cpInfo.queueFamilyIndex = qFamilyIndices.graphicsFamily; 
   cpInfo.flags = 0; 

if (vkCreateCommandPool(VulkanContext::getInstance()->
   getDevice()->logicalDevice, &cpInfo, nullptr, &commandPool) != 
   VK_SUCCESS) { 
         throw std::runtime_error(" failed to create 
            command pool !!"); 
   } 

   // Allocate command buffer and start recording 
   VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool); 

   //-- Copy the buffer 
   VkBufferCopy copyregion = {}; 
   copyregion.srcOffset = 0; 
   copyregion.dstOffset = 0; 
   copyregion.size = size; 
   vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer,
      1, &copyregion); 

   // End recording and Execute command buffer and free command buffer 
   endSingleTimeCommands(commandBuffer, commandPool); 

   vkDestroyCommandPool(VulkanContext::getInstance()->
      getDevice()->logicalDevice, commandPool, 
      nullptr); 

} 

```

In the function, we first get the queue family indices from the device. We then create a new command pool, and then we create a new command buffer using the `beginSingleTimeCommands` function. To copy the buffer, we create the `VkBufferCopy` struct. We set the source and destination offset to be `0` and set the buffer size.

To actually copy the buffers, we call the `vlCmdCopyBuffer` function, which takes in a command buffer, the source command buffer, the destination command buffer, the copy region count (which is `1` in this case), and the copy region struct. Once the buffers are copied, we call `endSingleTimeCommands` to destroy the command buffer and call `vkDestroyCommandPool` to destroy the command pool itself.

Now, we can go back to the `createVertexBuffers` function in `ObjectsBuffers` and copy the staging buffer to the vertex buffer. We also destroy the staging buffer and the buffer memory:

```cpp
   vkTools::copyBuffer(stagingBuffer, 
         vertexBuffer, 
         bufferSize); 

   vkDestroyBuffer(VulkanContext::getInstance()->
      getDevice()->logicalDevice, stagingBuffer, nullptr); 
   vkFreeMemory(VulkanContext::getInstance()->
      getDevice()->logicalDevice, stagingBufferMemory, nullptr); 
```

The index buffers are created the same way, using the `createIndexBuffer` function:

```cpp

void ObjectBuffers::createIndexBuffer() { 

   VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size(); 

   VkBuffer stagingBuffer; 
   VkDeviceMemory stagingBufferMemory; 

   vkTools::createBuffer(bufferSize, 
       VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
       stagingBuffer, stagingBufferMemory); 

   void* data; 
   vkMapMemory(VulkanContext::getInstance()->
     getDevice()->logicalDevice, stagingBufferMemory, 
     0, bufferSize, 0, &data); 
   memcpy(data, indices.data(), (size_t)bufferSize); 
   vkUnmapMemory(VulkanContext::getInstance()->
     getDevice()->logicalDevice, stagingBufferMemory); 

   vkTools::createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |    
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
        indexBuffer,  
        indexBufferMemory); 

   vkTools::copyBuffer(stagingBuffer, 
         indexBuffer, 
         bufferSize); 

   vkDestroyBuffer(VulkanContext::getInstance()->
     getDevice()->logicalDevice, 
     stagingBuffer, nullptr); 
   vkFreeMemory(VulkanContext::getInstance()->
      getDevice()->logicalDevice, 
      stagingBufferMemory, nullptr); 

}    
```

Creating `UniformBuffer` is easier, because we will just be using the `HOST_VISIBLE` GPU memory, so staging buffers are not required:

```cpp
void ObjectBuffers::createUniformBuffers() { 

   VkDeviceSize bufferSize = sizeof(UniformBufferObject); 

   vkTools::createBuffer(bufferSize, 
               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
               uniformBuffers, uniformBuffersMemory); 

} 
```

Finally, we destroy the buffers and memories in the `destroy` function:

```cpp
void ObjectBuffers::destroy(){ 

   vkDestroyBuffer(VulkanContext::getInstance()->
     getDevice()->logicalDevice, uniformBuffers, nullptr); 
   vkFreeMemory(VulkanContext::getInstance()->
      getDevice()->logicalDevice, uniformBuffersMemory, 
      nullptr); 

   vkDestroyBuffer(VulkanContext::getInstance()->
     getDevice()->logicalDevice, indexBuffer, nullptr); 
   vkFreeMemory(VulkanContext::getInstance()->
      getDevice()->logicalDevice, indexBufferMemory, 
      nullptr); 

   vkDestroyBuffer(VulkanContext::getInstance()->
     getDevice()->logicalDevice, vertexBuffer, nullptr); 
   vkFreeMemory(VulkanContext::getInstance()->
     getDevice()->logicalDevice, vertexBufferMemory, nullptr); 

} 

```

# Creating the Descriptor class

Unlike OpenGL, where we had uniform buffers to pass in the model, view, projection, and other kinds of data, Vulkan has descriptors. In descriptors, we have to first specify the layout of the buffer, as well as the binding location, count, type of descriptor, and the shader stage it is associated with.

Once the descriptor layout is created with the different types of descriptors, we have to create a descriptor pool for the number-swapchain image count because the uniform buffer will be set for each time per frame.

After that, we can allocate and populate the descriptor sets for both the frames. The allocation of data will be done from the pool.

We will create a new class for creating the descriptor set, layout binding, pool, and allocating and populating the descriptor sets. Create a new class called `Descriptor`. In the `Descriptor.h` file, add the following code:

```cpp
#pragma once 
#include <vulkan\vulkan.h> 
#include <vector> 

class Descriptor 
{ 
public: 
   Descriptor(); 
   ~Descriptor(); 

   // all the descriptor bindings are combined into a single layout 

   VkDescriptorSetLayout descriptorSetLayout;  
   VkDescriptorPool descriptorPool; 
   VkDescriptorSet descriptorSet; 

   void createDescriptorLayoutSetPoolAndAllocate(uint32_t 
      _swapChainImageCount); 
   void populateDescriptorSets(uint32_t _swapChainImageCount, 
      VkBuffer uniformBuffers); 

   void destroy(); 

private: 

   void createDescriptorSetLayout(); 
   void createDescriptorPoolAndAllocateSets(uint32_t 
      _swapChainImageCount); 

};  

```

We include the usual `Vulkan.h` and vector. In the public section, we create the class with the constructor and the destructor. We also create three variables, called `descriptorSetLayout`, `descriptorPool`, and `descriptorSets`, of the `VkDescriptorSetLayout`, `VkDescriptorPool`, and `VkDescriptorSet` types for easy access to the set. The `createDescriptorLayoutSetPoolAndAllocate` function will call the private `createDescriptorSetLayout` and `createDescriptorPoolAndAllocateSets` functions, which will create the layout set and then create the descriptor pool and allocate to it. The `populateDescriptorSets` function will be called when we set the uniform buffer to populate the sets with the data.

We also have a `destroy` function to destroy the Vulkan objects that have been created. In the `Descriptor.cpp` file, we will add the implementations of the functions. Add the necessary includes first, and then add the constructor, destructor, and the `createDescriptorLayoutAndPool` function:

```cpp
#include "Descriptor.h" 

#include<array> 
#include "VulkanContext.h" 

#include "Mesh.h" 

Descriptor::Descriptor(){ 

} 
Descriptor::~Descriptor(){ 

} 

void Descriptor::createDescriptorLayoutSetPoolAndAllocate(uint32_t 
    _swapChainImageCount){ 

   createDescriptorSetLayout(); 
   createDescriptorPoolAndAllocateSets(_swapChainImageCount); 

} 

```

The `createDescriptorLayoutSetPoolAndAllocate` function calls the `createDescriptorSetLayout` and `createDescriptorPoolAndAllocateSets` functions. Let's now add the `createDescriptorSetLayout` function:

```cpp
void Descriptor::createDescriptorSetLayout() { 

   VkDescriptorSetLayoutBinding uboLayoutBinding = {}; 
   uboLayoutBinding.binding = 0;// binding location 
   uboLayoutBinding.descriptorCount = 1; 
   uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;  
   uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;  

   std::array<VkDescriptorSetLayoutBinding, 1> 
      layoutBindings = { uboLayoutBinding }; 

   VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {}; 
   layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR*SET
*                            LAYOUT_CREATE_INFO; 
   layoutCreateInfo.bindingCount = static_cast<uint32_t> 
                                   (layoutBindings.size()); 
   layoutCreateInfo.pBindings = layoutBindings.data();  

   if (vkCreateDescriptorSetLayout(VulkanContext::getInstance()->
     getDevice()->logicalDevice, &layoutCreateInfo, nullptr, 
     &descriptorSetLayout) != VK_SUCCESS) { 

         throw std::runtime_error("failed to create 
           descriptor set layout"); 
   } 
} 
```

For our project, the layout set will just have one layout binding, which is the one struct with the model, view, and projection matrix information.

We have to populate the `VkDescriptorSetLayout` struct and specify the binding location index, the count, the type of information we will be passing in, and the shader stage to which the uniform buffer will be sent. After creating the set layout, we populate `VkDescriptorSetLayoutCreateInfo`, in which we specify the binding count and the bindings itself.

Then, we call the `vkCreateDescriptorSetLayout` function to create the descriptor set layout by passing in the logical device and the layout creation info. Next, we add the `createDescriptorPoolAndAllocateSets` function:

```cpp
void Descriptor::createDescriptorPoolAndAllocateSets(uint32_t 
    _swapChainImageCount) { 

   // create pool 
   std::array<VkDescriptorPoolSize, 1> poolSizes = {}; 

   poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
   poolSizes[0].descriptorCount = _swapChainImageCount; 

   VkDescriptorPoolCreateInfo poolInfo = {}; 
   poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; 
   poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());  
   poolInfo.pPoolSizes = poolSizes.data(); 

   poolInfo.maxSets = _swapChainImageCount;  

   if (vkCreateDescriptorPool(VulkanContext::getInstance()->
      getDevice()->logicalDevice, &poolInfo, nullptr, 
      &descriptorPool) != VK_SUCCESS) { 

         throw std::runtime_error("failed to create descriptor pool "); 
   } 

   // allocate 
   std::vector<VkDescriptorSetLayout> layouts(_swapChainImageCount, 
       descriptorSetLayout); 

   VkDescriptorSetAllocateInfo allocInfo = {}; 
   allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
   allocInfo.descriptorPool = descriptorPool; 
   allocInfo.descriptorSetCount = _swapChainImageCount; 
   allocInfo.pSetLayouts = layouts.data(); 

   if (vkAllocateDescriptorSets(VulkanContext::getInstance()->
      getDevice()->logicalDevice, &allocInfo, &descriptorSet) 
      != VK_SUCCESS) { 

         throw std::runtime_error("failed to allocate descriptor
            sets ! "); 
   }  
}
```

To create the descriptor pool, we have to specify the pool size using `VkDescriptorPoolSize`. We create an array of it and call it `poolSizes`. Since, in the layout set, we just have the uniform buffer, we set its type and set the count equal to the swap-chain-image count. To create the descriptor pool, we have to specify the type, pool-size count, and the pool-size data. We also have to set the maxsets, which is the maximum number of sets that can be allocated from the pool, which is equal to the swap-chain-image count. We create the descriptor pool by calling `vkCreateDescriptorPool` and passing in the logical device and the pool-creation info. Next, we have to specify allocation parameters for the description sets.

We create a vector of the descriptor set layout. Then, we create the `VkDescriptionAllocationInfo` struct to populate it. We pass in the description pool, the descriptor set count (which is equal to the swap-chain-images count), and pass in the layout data. Then, we allocate the descriptor sets by calling `vkAllocateDescriptorSets` and passing in the logical device and the create info struct.

Finally, we will add the `populateDescriptorSets` function, as follows:

```cpp
void Descriptor::populateDescriptorSets(uint32_t _swapChainImageCount, 
   VkBuffer uniformBuffers) { 

   for (size_t i = 0; i < _swapChainImageCount; i++) { 

         // Uniform buffer info 

         VkDescriptorBufferInfo uboBufferDescInfo = {}; 
         uboBufferDescInfo.buffer = uniformBuffers; 
         uboBufferDescInfo.offset = 0; 
         uboBufferDescInfo.range = sizeof(UniformBufferObject); 

         VkWriteDescriptorSet uboDescWrites; 
         uboDescWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; 
         uboDescWrites.pNext = NULL; 
         uboDescWrites.dstSet = descriptorSet; 
         uboDescWrites.dstBinding = 0; // binding index of 0  
         uboDescWrites.dstArrayElement = 0;  
         uboDescWrites.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
         uboDescWrites.descriptorCount = 1;  
         uboDescWrites.pBufferInfo = &uboBufferDescInfo; // uniforms 
                                                            buffers 
         uboDescWrites.pImageInfo = nullptr; 
         uboDescWrites.pTexelBufferView = nullptr; 

         std::array<VkWriteDescriptorSet, 1> descWrites = { uboDescWrites}; 

         vkUpdateDescriptorSets(VulkanContext::getInstance()->
           getDevice()->logicalDevice, static_cast<uint32_t>
           (descWrites.size()), descWrites.data(), 0, 
           nullptr); 
   } 

} 
```

This function takes in the swapchain image count and the uniform buffer as parameters. For both the images of the swapchain, the configuration of the descriptor needs to be updated by calling `vkUpdateDescriptorSets`. This function takes in an array of `VkWriteDescriptorSet`. Now, `VkWriteDescriptorSet` takes in either a buffer, image struct, or `texelBufferView` as a parameter. Since we are going to use the uniform buffer, we will have to create it and pass it in. `VkdescriptorBufferInfo` takes in a buffer (which will be the uniform buffer we created), takes an offset (which is none in this case), and then takes the range (which is the size of the buffer itself).

After creating it, we can start specifying `VkWriteDescriptorSet`. This takes in the type, `descriptorSet`, and the binding location (which is the 0^(th) index). It has no array elements in it, and it takes the descriptor type (which is the uniform buffer type); the descriptor count is `1`, and we pass in the buffer info struct. For the image info and the texel buffer view, we specify none, as it is not being used.

We then create an array of `VkWriteDescriptorSet` and add the uniform buffer descriptor writes info we created, called `uboDescWrites`, to it. We update the descriptor set by calling `vkUpdateDescriptorSets` and pass in the logical device, the descriptor writes size, and the data. That's it for the `populateDescriptorSets` function. We finally add the destroy function, which destroys the descriptor pool and the descriptor set layout. Add the function as follows:

```cpp
void Descriptor::destroy(){ 

   vkDestroyDescriptorPool(VulkanContext::getInstance()->
      getDevice()->logicalDevice, descriptorPool, nullptr); 
   vkDestroyDescriptorSetLayout(VulkanContext::getInstance()->
      getDevice()->logicalDevice, descriptorSetLayout, nullptr); 
} 
```

# Creating the SPIR-V shader binary

Unlike OpenGL, which takes in GLSL (OpenGL Shading Language) human-readable files for shaders, Vulkan takes in shaders in binary or byte code format. All shaders, whether vertex, fragment, or compute, have to be in byte code format.

SPIR-V is also good for cross-compilation, making porting shader files a lot easier. If you have a Direct3D HLSL shader code, it can be compiled to SPIR-V format and can be used in a Vulkan application, making it very easy to port Direct3D games to Vulkan. The shader is initially written in GLSL, with some minor changes to how we wrote it for OpenGL. A compiler is provided, which compiles the code from GLSL to SPIR-V format. The compiler is included with the Vulkan SDK installation. The basic vertex-shader GLSL code is as follows:

```cpp
#version 450 
#extension GL_ARB_separate_shader_objects : enable 

layout (binding = 0) uniform UniformBufferOBject{ 

mat4 model; 
mat4 view; 
mat4 proj; 

} ubo; 

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inNormal; 
layout(location = 2) in vec3 inColor; 
layout(location = 3) in vec2 inTexCoord; 

layout(location = 0) out vec3 fragColor; 

void main() { 

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0); 
    fragColor = inColor; 

}
```

The shader should look very familiar, with some minor changes. For example, the GLSL version is still specified at the top. In this case, it is `#version 450`. But we also see new things, such as `#extension GL_ARB_seperate_shader_objects: enable`. This specifies the extension that the shader uses. In this case, an old extension is needed, which basically lets us use vertex and fragment shaders as separate files. Extensions need to be approved by the **Architecture Review Board (ARB).**

Apart from the inclusion of the extension, you may have noticed that there is a location layout specified for all data types. And you may have noticed that when creating vertex and uniform buffers, we had to specify the binding index for the buffers. In Vulkan, there is no equivalent to `GLgetUniformLocation` to get the location index of a uniform buffer. This is because it takes quite a bit of system resources to get the location. Instead, we specify and sort of hardcode the value of the index. The uniform buffers, as well as  the in and out buffer can all be assigned an index of 0  as they are of different data types. Since the uniform buffer will be sent as a st ruct with model, view, and projection matrices, a similar struct is created in the shader and assigned to the 0^(th) index of the uniform layout.

All four attributes are also assigned a layout index `0`, `1`, `2`, and `3`, as specified in the `Mesh.h` file under the `Vertex` struct when setting the `VkVertexInputAttributeDescription` for the four attributes. The out is also assigned a layout location index of `0`, and the data type is specified as `vec3`. Then, in the main function of the shader, we set the `gl_Position` value by multiplying the local coordinate of the object by the model, view, and projection matrix received from the uniform buffer struct. Furthermore, `outColor` is set as `inColor` received.

The fragment shader is as follows: open a `.txt` file, add the shader code to it, and name the file `basic.` Also, change the extension to `*.vert` from `*.txt`:

```cpp
#version 450 
#extension GL_ARB_separate_shader_objects : enable 

layout(location = 0) in vec3 fragColor; 

layout(location = 0) out vec4 outColor; 

void main() { 

   outColor = vec4(fragColor, 1.0f); 

}
```

Here, we specify the GLSL version and the extension to use. There are `in` and `out`, both of which have a location layout of `0`. Note that `in` is a `vec3` called `fragColor`, which is what we sent out of the vertex shader, and that `outColor` is a `vec4`. In the main function of the shader file, we convert `vec3` to `vec4` and set the resultant color to `outColor`. Add the fragment shader to a file called `basic.frag`. In the `VulkanProject` root directory, create a new folder called `shaders` and add the two shader files to it:

![](img/4fdcc29c-5465-47f9-ae6a-9cf9cb4e2a4a.png)

Create a new folder called `SPIRV`, as this is where we will put the compiled SPIRV bytecode file. To compile the `.glsl` files, we will use the `glslValidator.exe` file, which was installed when we installed the Vulkan SDK. Now, to compile the code, we can use the following command:

```cpp
glslangValidator.exe -V basic.frag -o basic.frag.spv 
```

Hold *Shift* on the keyboard, right-click in the `shaders` folder, and click the Open PowerShell window here:

![](img/7a0d7a83-d6ce-4d54-bf04-4b231e46c0eb.png)

In PowerShell, type in the following command:

![](img/955dd04b-e602-4fb5-8d39-d26bec5707d6.png)

Make sure the *V* is capitalized and the *o* is lowercase, otherwise, it will give compile errors. This will create a new `spirv` file in the folder. Change `frag` to `vert` to compile the SPIRV vertex shader:

![](img/bf23acba-9668-4343-ba9e-8b8d4d506b37.png)

This will create the vertex and fragment shader SPIRV binaries in the folder:

![](img/90596c8c-2d1a-4622-8dfa-a4c66b908fbb.png)

Instead of compiling the code manually each time, we can create a `.bat` file that can do this for us, and put the compiled SPIRV binaries in the `SPIRV` folder. In the `shaders` folder, create a new `.txt` file and name it `glsl_spirv_compiler.bat`.

In the `.bat` file, add the following:

```cpp
@echo off 
echo compiling glsl shaders to spirv  
for /r %%i in (*.vert;*.frag) do %VULKAN_SDK%\Bin32\glslangValidator.exe -V "%%i" -o  "%%~dpiSPIRV\%%~nxi".spv 
```

Save and close the file. Now double-click on the `.bat` file to execute it. This will compile the shaders and place the compiled binary in the SPIRV shader files:

![](img/d4acaddc-c373-4a63-934f-4d751d123877.png)

You can delete the SPIRV files in the `shaders` folder that we compiled earlier using the console command because we will be using the shader files in the `SPIRV` subfolder.

# Summary

In this chapter, we created all the resources required to render the geometry. First of all, we added the `Mesh` class, which has vertex and index information for all the mesh types, including triangle, quad, cube, and sphere. Then, we created the `ObjectBuffers` class, which was used to store and bind the buffers to the GPU memory using the `VkTool` file. We also created a separate descriptor class, which has our descriptor set layout and pool. In addition, we created descriptor sets. Finally, we created the SPIRV bytecode shader files, which were compiled from the GLSL shader.

In the next chapter, we will use the resources we created here to draw our first colored geometry.