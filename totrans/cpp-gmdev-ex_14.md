# Preparing the Clear Screen

In the last chapter, we enabled the Vulkan validation layers and extensions, created the Vulkan application and instance, chose the device, and created the logical device. In this chapter, we will continue the journey toward creating a clear screen picture and presenting it to the viewport.

Before drawing pictures, we first clear and erase the previous picture with a color value. If we don't do this, the new picture will be written on top of the previous picture, which will create a psychedelic effect.

Each picture is cleared and rendered and then presented on the screen. While the current one is being shown, the next picture is in the process of being drawn in the background. Once that is rendered, the current picture will be swapped with the new picture. This swapping of pictures is taken care of by the **SwapChain**.

In our case, each picture in the SwapChain we are drawing simply stores the color information. This target picture is rendered and, hence, it is called the render target. We can also have other target pictures. For example, we can have a depth target/picture, which will store the depth information of each pixel per frame. Consequently, we create these render targets as well.

Each of these target pictures per frame is set as an attachment and used to create the framebuffer. Since we have double buffering (meaning we have two sets of pictures to swap between), we create a framebuffer for each frame. Consequently, we will create two framebuffers—one for each frame—and we will add the picture as an attachment.

The commands that we give to the GPU—for example, the draw command—are sent to the GPU with each frame using a command buffer. The command buffer stores all the commands to be submitted to the GPU using the graphics queue of the device. So, for each frame, we create a command buffer to carry all our commands as well.

Once the commands are submitted and the scene is rendered, instead of presenting the drawn picture to the screen, we can save it and add any post-processing effects to it, such as motion blur. In the renderpass, we can specify how the render target is to be used. Although in our case, we are not going to add any post-processing effects, we still need to create a renderpass. Consequently, we create a renderpass, which will specify to the device how many swap chain pictures and buffers we will be using, what kind of buffers they are, and how they are to be used.

The different stages the picture will go through are as follows:

![](img/42991319-7f62-4966-8aee-afd84dc7cc22.png)

The topics covered in this chapter are as follows:

*   Creating `SwapChain`
*   Creating `Renderpass`
*   Using render views and `Framebuffers`
*   Creating `CommandBuffer`
*   Beginning and ending `Renderpass`
*   Creating the clear screen

# Creating SwapChain

While the scene is rendered, the buffers are swapped and presented to the window surface. The surface is platform-dependent and, depending upon the operating system, we have to choose the surface format accordingly. For the scene to be presented properly, we create the `SwapChain`, depending upon the surface format, presentation mode, and the extent, meaning the width and height, of the picture that the window can support.

In [Chapter 10](dc30df72-df2e-4bb9-a598-a481ecb595f3.xhtml), *Drawing Vulkan Objects*, when we chose the GPU device to use, we retrieved the properties of the device, such as the surface format and the presentation modes it supports. While we create the `SwapChain`, we match and check the surface format and the presentation that is available from the device, and that is also supported by the window to create the `SwapChain` object itself.

We create a new class called `SwapChain` and add the following includes to `SwapChain.h`:

```cpp
#include <vulkan\vulkan.h> 
#include <vector> 
#include <set> 
#include <algorithm>  
```

We then create the class, as follows:

```cpp
classSwapChain { 
public: 
   SwapChain(); 
   ~SwapChain(); 

   VkSwapchainKHR swapChain; 
   VkFormat swapChainImageFormat; 
   VkExtent2D swapChainImageExtent; 

   std::vector<VkImage> swapChainImages; 

   VkSurfaceFormatKHRchooseSwapChainSurfaceFormat(
     const std::vector<VkSurfaceFormatKHR>&availableFormats); 

   VkPresentModeKHRchooseSwapPresentMode(
     const std::vector<VkPresentModeKHR>availablePresentModes); 

   VkExtent2DchooseSwapExtent(constVkSurfaceCapabilitiesKHR&capabilities); 

   void create(VkSurfaceKHRsurface); 

void destroy(); 
}  
```

In the public section of the class, we create the constructor and the destructor. Then, we create the variables of the `VkSwapchainKHR`, `VkFormat`, and `VkExtent2D` types to store the `swapchain` itself. When we create the surface, we store the format of the picture itself, which is supported, as well as the extent of the picture, which is the width and height of the viewport. This is because, when the viewport is stretched or changed, the size of the swapchain picture will also be changed accordingly.

We create a vector of the `VkImage` type, called `swapChainImages`, to store the SwapChain pictures. Three helper functions, `chooseSwapChainSurfaceFormat`,  `chooseSwapPresentMode`, and `chooseSwapExtent`, are created to get the most suitable surface format, present mode, and SwapChain extent. Finally, the `create` function takes the surface in which we will create the swapchain itself. We also add a function to destroy and release the resources back to the system.

That is it for the `SwapChain.h` file. We will now move on to `SwapChain.cpp` to incorporate the implementation of the functions.

In the `SwapChain.cpp` file, add the following includes:

```cpp
#include"SwapChain.h" 

#include "VulkanContext.h"
```

We will need to include `VulkanContext.h`  to get the device's `SwapChainSupportDetails` struct, which we populated in the last chapter when we selected the physical device and created the logical device. Before we create the swapchain, let's first look at the three helper functions and see how each is created.

The first of the three functions is `chooseSwapChainSurfaceFormat`. This function takes in a vector of `VkSurfaceFormatKHR`, which is the available format supported by the device. Using this function, we will choose the surface format that is most suitable. The function is created as follows:

```cpp
VkSurfaceFormatKHRSwapChain::chooseSwapChainSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&availableFormats) { 

   if (availableFormats.size() == 1 &&availableFormats[0].format == 
     VK_FORMAT_UNDEFINED) { 
         return{VK_FORMAT_B8G8R8A8_UNORM, 
           VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }; 
   } 

   for (constauto& availableFormat : availableFormats) { 
         if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM&& 
            availableFormat.colorSpace == 
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { 
                return availableFormat; 
         } 
   } 
   returnavailableFormats[0]; 
} 
```

First, we check whether the available format is just `1` and whether it is undefined by the device. This means that there is no preferred format, so we choose the one that is most convenient for us.

The values returned are the color format and color space. The color format specifies the format of the color itself, `VK_FORMAT_B8G8R8A8_UNORM`, which tells us that we store 32 bits of information in each pixel. The colors are stored in the Blue, Green, Red, and Alpha channels, and in that order. Each channel is stored in 8 bits, so that means 2^8, which is 256 color values. `UNORM` suggests that each color value is normalized, so the color values, instead of being from 0-255, are normalized between 0 and 1.

We choose the SRGB color space as the second parameter, as we want more of a range of colors to be represented. If there is no preferred format, we go through the formats available and then check and return the ones that we need. We choose this color space, as most surfaces support this format because it is widely available. Otherwise, we just return the first available format.

The next function is `chooseSwapPresentMode`, which takes in a vector of `VkPresentModeKHR` called `availablePresentModes`. Presentation modes specify how the final rendered picture is presented to the viewport. Here are the available modes:

*   `VK_PRESENT_MODE_IMMEDIATE_KHR`: In this case, the picture will be displayed as soon as a picture is available to present. Pictures are not queued to be displayed. This causes picture tearing.

*   `VK_PRESENT_MODE_FIFO_KHR`: The acquired pictures to be presented are put in a queue. The size of the queue is one minus the swap chain size. At vsync, the first picture to be displayed gets displayed in the **First In First Out (FIFO)** manner. There is no tearing as pictures are displayed in the same order in which they were added to the queue and vsync is enabled. This mode needs to always be supported.
*   `VK_PRESENT_MODE_FIFO_RELAXED_KHR`: This is a variation of the `FIFO` mode. In this mode, if the rendering is faster than the refresh rate of the monitor, it is fine, but if the drawing is slower than the monitor, there will be screen tearing as the next available picture is presented immediately.
*   `VK_PRESENTATION_MODE_MAILBOX_KHR`: The presentation of the pictures is put in a queue, but it has just one element in it, unlike `FIFO`, which has more than one element in the queue. The next picture to be displayed will wait for the queue to be displayed and then the presentation engine will display the picture. This doesn't cause tearing.

With this information, let's create the `chooseSwapPresentMode` function:

```cpp
VkPresentModeKHRSwapChain::chooseSwapPresentMode(
  const std::vector<VkPresentModeKHR>availablePresentModes) { 

   VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR; 

   for (constauto& availablePresentMode : availablePresentModes) { 

         if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) { 
               return availablePresentMode; 
         } 
         elseif (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { 
               bestMode = availablePresentMode; 
         } 

         return bestMode; 
   } 
} 
```

Since the `FIFO` mode is our most preferred mode, we set it in the function so that we can compare it with the available modes of the device. If it is not available, we go for the next best mode, which is the `MAILBOX` mode, so that the presentation queue will have at least one more picture to avoid screen tearing. If neither mode is available, we go for the `IMMEDIATE` mode, which is least desirable.

The third function is the `chooseSwapExtent` function. In this function, we get the resolution of the window that we drew to set the resolution of the swapchain pictures. It is added as follows:

```cpp

VkExtent2DSwapChain::chooseSwapExtent(constVkSurfaceCapabilitiesKHR&
   capabilities) { 

   if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) { 
         returncapabilities.currentExtent; 
   } 
   else { 

         VkExtent2D actualExtent = { 1280, 720 }; 

         actualExtent.width = std::max(capabilities.minImageExtent.
                              width, std::min(capabilities.
                              maxImageExtent. width, 
                              actualExtent.width)); 
         actualExtent.height = std::max(capabilities.minImageExtent.
                               height, std::min(capabilities.
                               maxImageExtent.height, 
                               actualExtent.height)); 

         return actualExtent; 
   } 
} 

```

The resolution of this window should match the swapchain pictures. Some window managers allow the resolution to be different between the pictures and the window. This is indicated by setting the value to the maximum of `uint32_t`. If not, then in that case, we return the current extent that we retrieved by the capabilities of the hardware, or pick the resolution that best matches the resolution between the maximum and minimum values available, as compared to the actual resolution we set, which is 1,280 x 720.

Let's now look at the `create` function, in which we actually create the `SwapChain` itself. To create this function, we will add the functionality to create the `SwapChain`:

```cpp
void SwapChain::create(VkSurfaceKHR surface) { 
... 
} 

```

The first thing we do is get the device support details, which we retrieved for our device when we created the `Device` class:

```cpp
SwapChainSupportDetails swapChainSupportDetails = VulkanContext::getInstance()-> getDevice()->swapchainSupport;
```

Then, using the `helper` function we created, we get the surface format, present mode, and the extent:

```cpp
   VkSurfaceFormatKHR surfaceFormat = chooseSwapChainSurfaceFormat
     (swapChainSupportDetails.surfaceFormats); 
   VkPresentModeKHR presentMode = chooseSwapPresentMode
     (swapChainSupportDetails.presentModes); 
   VkExtent2D extent = chooseSwapExtent
      (swapChainSupportDetails.surfaceCapabilities); 
```

We then set the minimum number of pictures required to make the swapchain:

```cpp
uint32_t imageCount = swapChainSupportDetails.
                      surfaceCapabilities.minImageCount; 
```

We should also make sure that we don't exceed the maximum available picture count, so if the `imageCount` is more than the maximum amount, we set `imageCount` to the maximum count:

```cpp
   if (swapChainSupportDetails.surfaceCapabilities.maxImageCount > 0 && 
     imageCount > swapChainSupportDetails.surfaceCapabilities.
     maxImageCount) { 
         imageCount = swapChainSupportDetails.surfaceCapabilities.
                      maxImageCount; 
   } 

```

To create the swapchain, we have to populate the `VkSwapchainCreateInfoKHR` struct first, so let's create it. Create a variable called `createInfo` and specify the type of the structure:

```cpp
   VkSwapchainCreateInfoKHR createInfo = {}; 
   createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR; 
```

In this, we have to specify the surface to use, the minimum picture count, the picture format, the space, and the extent. We also need to specify the picture array layers. Since we are not going to create a stereoscopic application like a virtual reality game, in which there would be two surfaces, one for the left eye and one for the right eye, instead, we just set the value for it as `1`. We also need to specify what the picture will be used for. Here, it will be used to show the color information using the color attachment:

```cpp
   createInfo.surface = surface; 
   createInfo.minImageCount = imageCount; 
   createInfo.imageFormat = surfaceFormat.format; 
   createInfo.imageColorSpace = surfaceFormat.colorSpace; 
   createInfo.imageExtent = extent; 
   createInfo.imageArrayLayers = 1; // this is 1 unless you are making
   a stereoscopic 3D application 
   createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
```

We now specify the graphics, presentation indices, and the count. We also specify the sharing mode. It is possible for the presentation and graphics family to be either the same or different.

If the presentation and graphics family is different, the sharing mode is said to be of the `VK_SHARING_MODE_CONCURRENT` type. This means that the picture can be used across multiple queue families. However, if the picture is in the same queue family, the sharing mode is said to be of the `VK_SHARING_MODE_EXCLUSIVE` type:

```cpp
   if (indices.graphicsFamily != indices.presentFamily) { 

         createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; 
         createInfo.queueFamilyIndexCount = 2; 
         createInfo.pQueueFamilyIndices = queueFamilyIndices; 

   } 
   else { 

         createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; 
         createInfo.queueFamilyIndexCount = 0; 
         createInfo.pQueueFamilyIndices = nullptr; 
   } 
```

If we want, we can apply a pre-transform to the picture to either flip it or mirror it. In this case, we just keep the current transform. We can also alpha-blend the picture with other window systems, but we just keep it opaque and ignore the alpha channel, set the present mode, and set whether the pixel should be clipped if there is a window in front. We can also specify an old `SwapChain` if the current one becomes invalid when we resize the window. Since we don't resize the window, we don't have to specify an older swapchain.

After setting the info struct, we can create the swapchain itself:

```cpp
if (vkCreateSwapchainKHR(VulkanContext::getInstance()->getDevice()->
   logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) { 
         throw std::runtime_error("failed to create swap chain !"); 
   } 
```

We create the swapchain using the `vkCreateSwapchainKHR` function, which takes the logical device, the `createInfo` struct, an allocator callback, and the swapchain itself. If it doesn't create the `SwapChain` because of an error, we will send out an error. Now that the swapchain is created, we will obtain the swapchain pictures.

Depending upon the picture count, we call the `vkGetSwapchainImagesKHR` function, which we use to first get the picture count and then call the function again to populate the `vkImage` vector with the pictures:

```cpp
   vkGetSwapchainImagesKHR(VulkanContext::getInstance()->getDevice()->
      logicalDevice, swapChain, &imageCount, nullptr); 
   swapChainImages.resize(imageCount); 
   vkGetSwapchainImagesKHR(VulkanContext::getInstance()->getDevice()->
      logicalDevice, swapChain, &imageCount, swapChainImages.data()); 
```

The creation of pictures is a bit more involved, but Vulkan automatically creates color pictures. We can set the picture format and extent as well:

```cpp
   swapChainImageFormat = surfaceFormat.format; 
   swapChainImageExtent= extent; 
```

Then, we add the `destroy` function, which destroys the `SwapChain` by calling the `vkDestroySwapchainKHR` function:

```cpp
void SwapChain::destroy(){ 

   // Swapchain 
   vkDestroySwapchainKHR(VulkanContext::getInstance()-> getDevice()->
      logicalDevice, swapChain, nullptr); 

} 
```

In the `VulkanApplication.h` file, include the `SwapChain` header and create a new `SwapChain` instance in the `VulkanApplication` class. In `VulkanApplication.cpp`, in the `initVulkan` function, after creating the logical device, create the `SwapChain` as follows:

```cpp
   swapChain = new SwapChain(); 
   swapChain->create(surface);
```

Build and run the application to make sure the `SwapChain` is created without any errors.

# Creating Renderpass

After creating the `SwapChain`, we move on to the Renderpass. Here, we specify how many color attachments and depth attachments are present and how many samples to use for each of them for each framebuffer.

As mentioned at the start of this chapter, a framebuffer is a collection of target attachments. Attachments can be of type color, depth, and so on. The color attachment stores the color information that is presented to the viewport. There are other attachments that the end user doesn't see, but that are used internally. This includes depth, for example, which has all the depth information per pixel. In the render pass, apart from the type of attachments, we also specify how the attachments are used.

For this book, we will be presenting what is rendered in a scene to the viewport, so we will just use a single pass. If we add a post-processing effect, we will take the rendered picture and apply this effect to it, for which we will need to use multiple passes. We will create a new class called `Renderpass`, in which we will create the render pass.

In the `Renderpass.h` file, add the following includes and class:

```cpp
#include <vulkan\vulkan.h> 
#include <array> 

class Renderpass 
{ 
public: 
   Renderpass(); 
   ~Renderpass(); 

   VkRenderPass renderPass; 

   void createRenderPass(VkFormat swapChainImageFormat); 

   void destroy(); 
};
```

In the class, add the constructor, destructor, and the `VkRenderPass` and `renderPass` variables. Add a new function called `createRenderPass` to create the `Renderpass` itself, which takes in the picture format. Also, add a function to destroy the `Renderpass` object after use.

In the `Renderpass.cpp` file, add the following includes, as well as the constructor and destructor:

```cpp
#include"Renderpass.h" 
#include "VulkanContext.h"
Renderpass::Renderpass(){} 

Renderpass::~Renderpass(){} 

```

We now add the `createRenderPass` function, in which we will add the functionality to create the Renderpass for the current scene to be rendered:

```cpp
voidRenderpass::createRenderPass(VkFormatswapChainImageFormat) { 
... 
} 
```

When we create the render pass, we have to specify the number and the type of attachments that we are using. So, for our project, we want only color attachments as we will only be drawing color information. We could also have a depth attachment, which stores depth information. We need to provide subpasses, and if so, then how many, as we could be using subpasses for adding post-processing effects to the current frame.

For the attachments and subpasses, we have to populate structs to pass to them at the time of creating the render pass.

So, let's populate the structs. First, we create the attachments:

```cpp
   VkAttachmentDescription colorAttachment = {}; 
   colorAttachment.format = swapChainImageFormat; 
   colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; 
   colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; 
   colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;  
   colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; 
   colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; 

   colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
   colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  
```

We create the struct and specify the format to be used, which is the same as the `swapChainImage` format. We have to provide the sample count as 1 as we are not going to be using multi-sampling. In `loadOp` and `storeOp`, we specify what to do with the data before and after rendering. We specify that at the time of loading the attachment, we will clear the data to a constant at the start. After the render process, we store the data so we can read from it later. We then decide what to do with the data before and after the stencil operation.

Since we are not using the stencil buffer, we specify DON'T CARE during loading and storing. We also have to specify the data layout before and after processing the picture. The previous layout of the picture doesn't matter, but after rendering, the picture needs to be changed to the layout in order for it to be ready for presenting.

Now we'll go through the subpass. Each subpass references the attachments that need to be specified as a separate structure:

```cpp
   VkAttachmentReference colorAttachRef = {}; 
   colorAttachRef.attachment = 0;  
   colorAttachRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  

```

In the `subpass` reference, we specify the attachment index, which is the 0^(th) index and specify the layout, which is a color attachment with optimal performance. Next, we create the `subpass` structure:

```cpp
   VkSubpassDescription subpass = {}; 
   subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; 
   subpass.colorAttachmentCount = 1;  
   subpass.pColorAttachments = &colorAttachRef; 

```

In the pipeline bind point, we specify that this is a graphics subpass, as it could have been a compute subpass. Specify the attachment count as `1` and provide the color attachment. Now, we can create the renderpass info struct:

```cpp
   std::array<VkAttachmentDescription, 1> attachments = 
      { colorAttachment }; 

   VkRenderPassCreateInfo rpCreateInfo = {}; 
   rpCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO; 
   rpCreateInfo.attachmentCount = static_cast<uint32_t>
                                  (attachments.size()); 
   rpCreateInfo.pAttachments = attachments.data(); 
   rpCreateInfo.subpassCount = 1; 
   rpCreateInfo.pSubpasses = &subpass; 

```

We create an array of one element of the `VkAttachmentDescription` type, and then we create the info struct and pass in the type. The attachment count and the attachments are passed in, and then the subpass count and the subpass is passed in as well. Create the renderpass itself by calling `vkCreateRenderPass` and passing in the logical device, the create info, and the allocator callback to get the renderpass:

```cpp

 if (vkCreateRenderPass(VulkanContext::getInstance()-> 
   getDevice()->logicalDevice, &rpCreateInfo, nullptr, &renderPass)
   != VK_SUCCESS) { 
         throw std::runtime_error(" failed to create renderpass !!"); 
   }
```

Finally, in the `destroy` function, we call `vkDestroyRenderPass` to destroy it after we are done:

```cpp
voidRenderpass::destroy(){ 

   vkDestroyRenderPass(VulkanContext::getInstance()-> getDevice()->
      logicalDevice, renderPass, nullptr); 

} 
```

In `VulkanApplication.h`, include `RenderPass.h` and create a render pass object. In `VulkanApplication.cpp`, after creating the swapchain, create the renderpass:

```cpp
   renderPass = new Renderpass(); 
   renderPass->createRenderPass(swapChain->swapChainImageFormat); 
```

Now, build and run the project to make sure there are no errors.

# Using render targets and framebuffers

To use a picture, we have to create an `ImageView`. The picture doesn't have any information, such as mipmap levels, and you can't access a portion of the picture. However, by now using picture views, we specify the type of the texture and whether it has mipmaps. In addition, in renderpass, we specified the attachments per frame buffer. We will create framebuffers here and pass in the picture views as attachments.

Create a new class called `RenderTexture`. In the `RenderTexture.h` file, add the following headers and then create the class itself:

```cpp
 #include <vulkan/vulkan.h> 
#include<array> 

class RenderTexture 
{ 
public: 
   RenderTexture(); 
   ~RenderTexture(); 

   std::vector<VkImage> _swapChainImages; 
   VkExtent2D _swapChainImageExtent; 

   std::vector<VkImageView> swapChainImageViews; 
   std::vector<VkFramebuffer> swapChainFramebuffers; 

   void createViewsAndFramebuffer(std::vector<VkImage> swapChainImages,  
     VkFormat swapChainImageFormat, VkExtent2D swapChainImageExtent, 
     VkRenderPass renderPass); 

   void createImageViews(VkFormat swapChainImageFormat); 
   void createFrameBuffer(VkExtent2D swapChainImageExtent, 
      VkRenderPass renderPass); 

   void destroy(); 

}; 
```

In the class, we add the constructor and destructor as usual. We will store `swapChainImages` and the extent to use it locally. We create two vectors to store the created ImageViews and framebuffers. To create the views and framebuffers, we will call the `createViewsAndFramebuffers` function, which takes the pictures, picture format, extent, and the renderpass as the input. This function intern will call `createImageViews` and `CreateFramebuffer` to create the views and buffers. We will add the `destroy` function, which destroys and releases the resources back to the system.

In the `RenderTexture.cpp` file, we will add the following includes as well as the constructor and destructor:

```cpp
#include "RenderTexture.h" 
#include "VulkanContext.h" 
RenderTexture::RenderTexture(){} 

RenderTexture::~RenderTexture(){} 
```

Then, add the `createViewAndFramebuffer` function:

```cpp
void RenderTexture::createViewsAndFramebuffer(std::vector<VkImage> swapChainImages, VkFormat swapChainImageFormat,  
VkExtent2D swapChainImageExtent,  
VkRenderPass renderPass){ 

   _swapChainImages =  swapChainImages; 
   _swapChainImageExtent = swapChainImageExtent; 

   createImageViews(swapChainImageFormat); 
   createFrameBuffer(swapChainImageExtent, renderPass); 
}
```

We first assign the images and `imageExtent` to the local variables. Then, we call the `imageViews` function, followed by `createFramebuffer`, in order to create both of them. To create the image views, use the `createImageViews` function:

```cpp
void RenderTexture::createImageViews(VkFormat swapChainImageFormat){ 

   swapChainImageViews.resize(_swapChainImages.size()); 

   for (size_t i = 0; i < _swapChainImages.size(); i++) { 

         swapChainImageViews[i] = vkTools::createImageView
                                  (_swapChainImages[i], 
               swapChainImageFormat,  
               VK_IMAGE_ASPECT_COLOR_BIT); 
   } 
} 
```

We specify the vector size depending upon the swapchain image count first. For each of the image counts, we create image views using the `createImageView` function in the `vkTool` namespace. The `createImageView` function takes in the image itself, the image format, and `ImageAspectFlag`. This will be `VK_IMAGE_ASPECT_COLOR_BIT` or `VK_IMAGE_ASPECT_DEPTH_BIT` depending on the kind of view that you want to create for the image. The `createImageView` function is created in the `Tools.h` file under the `vkTools` namespace. The `Tools.h` file is as follows:

```cpp
#include <vulkan\vulkan.h> 
#include <stdexcept> 
#include <vector> 

namespace vkTools { 

   VkImageView createImageView(VkImage image, VkFormat format, 
       VkImageAspectFlags aspectFlags); 

}
```

The implementation of the function is created in the `Tools.cpp` file as follows:

```cpp
#include "Tools.h" 
#include "VulkanContext.h"

namespace vkTools { 
   VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) { 

         VkImageViewCreateInfo viewInfo = {}; 
         viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; 
         viewInfo.image = image; 
         viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; 
         viewInfo.format = format; 

         viewInfo.subresourceRange.aspectMask = aspectFlags; 
         viewInfo.subresourceRange.baseMipLevel = 0; 
         viewInfo.subresourceRange.levelCount = 1; 
         viewInfo.subresourceRange.baseArrayLayer = 0; 
         viewInfo.subresourceRange.layerCount = 1; 

         VkImageView imageView; 
         if (vkCreateImageView(VulkanContext::getInstance()->
            getDevice()->logicalDevice, &viewInfo, nullptr, &imageView) 
            != VK_SUCCESS) { 
               throw std::runtime_error("failed to create 
                 texture image view !"); 
         } 

         return imageView; 
   } 

} 
```

To create the `imageView`, we have to populate the `VkImageViewCreateInfo` struct and then use the `vkCreateImageView` function to create the view itself. To populate the view info, we specify the structure type, the picture itself, the view type, which is `VK_IMAGE_VIEW_TYPE_2D`, and a 2D texture, and then specify the format. We pass in the aspectFlags for the aspect mask. We create the image view without any mipmap level or layers, so we set them to `0`. We would only need multiple layers if we were making something like a VR game.

We then create an `imageView` of the `VkImage` type and create it using the `vkCreateImageView` function, which takes in the logical device, the view info struct, and then the picture view is created and returned. That's all for the Tools file.

We will use the Tools file and add more functions to it when we want functions that can be reused. Now, let's go back to the `RenderTexture.cpp` file and add in the function to create the framebuffer.

We will create framebuffers for each frame in the swapchain. `createFramebuffer` requires the picture extent and the renderpass itself:

```cpp
void RenderTexture::createFrameBuffer(VkExtent2D swapChainImageExtent, VkRenderPass renderPass){ 

   swapChainFramebuffers.resize(swapChainImageViews.size()); 

   for (size_t i = 0; i < swapChainImageViews.size(); i++) { 

         std::array<VkImageView, 2> attachments = { 
               swapChainImageViews[i] 
         }; 

         VkFramebufferCreateInfo fbInfo = {}; 
         fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO; 
         fbInfo.renderPass = renderPass;  
         fbInfo.attachmentCount = static_cast<uint32_t>
                                  (attachments.size());; 
         fbInfo.pAttachments = attachments.data();; 
         fbInfo.width = swapChainImageExtent.width; 
         fbInfo.height = swapChainImageExtent.height; 
         fbInfo.layers = 1;  

         if (vkCreateFramebuffer(VulkanContext::getInstance()->
            getDevice()->logicalDevice, &fbInfo, NULL, 
            &swapChainFramebuffers[i]) != VK_SUCCESS) { 

               throw std::runtime_error(" failed to create 
                  framebuffers !!!"); 
         } 
   } 
} 

```

For each frame that we create, the Framebuffer first populates the `framebufferInfo` struct and then calls `vkCreateFramebuffer` to create the Framebuffer itself. For each frame, we create a new info struct and specify the type of struct. We then pass the renderpass, the attachment count, and the attachment views, specify the width and height of the Framebuffer, and set the layers to `1`.

Finally, we create the framebuffer by calling the `vkCreateFramebuffer` function:

```cpp
void RenderTexture::destroy(){ 

   // image views 
   for (auto imageView : swapChainImageViews) { 

         vkDestroyImageView(VulkanContext::getInstance()->getDevice()->
            logicalDevice, imageView, nullptr); 
   } 

   // Framebuffers 
   for (auto framebuffer : swapChainFramebuffers) { 
         vkDestroyFramebuffer(VulkanContext::getInstance()->
            getDevice()->logicalDevice, framebuffer, nullptr); 
   } 

} 
```

In the `destroy` function, we destroy each of the picture views and framebuffers we created by calling `vkDestroyImageView` and `vkDestroyFramebuffer`. And that is all for the `RenderTexture` class.

In `VulkanApplication.h`, include `RenderTexture.h` and create an instance of it called `renderTexture` in the `VulkanApplication` class. In the `VulkanApplication.cpp` file, include the `initVulkan` function and create a new `RenderTexture`:

```cpp
   renderTexture = new RenderTexture(); 
   renderTexture->createViewsAndFramebuffer(swapChain->swapChainImages, 
         swapChain->swapChainImageFormat, 
         swapChain->swapChainImageExtent, 
         renderPass->renderPass); 
```

# Creating CommandBuffer

In Vulkan, the drawing and other operations done on the GPU are performed using command buffers. The command buffers contain the draw commands, which are recorded and then executed. Draw commands are to be recorded and executed in every frame. To create a command buffer, we have to first create a command pool and then allocate command buffers from the command pool, and then the commands are recorded per frame.

Let's create a new class for creating the command buffer pool and then allocate the command buffers. We also create a function to start and stop recording and to destroy the command buffers. Create a new class, called `DrawCommandBuffer`, and `DrawCommandBuffer.h` as follows:

```cpp
#include <vulkan\vulkan.h> 
#include <vector> 

class DrawCommandBuffer 
{ 
public: 
   DrawCommandBuffer(); 
   ~DrawCommandBuffer(); 

   VkCommandPool commandPool; 
   std::vector<VkCommandBuffer> commandBuffers; 

   void createCommandPoolAndBuffer(size_t imageCount); 
   void beginCommandBuffer(VkCommandBuffer commandBuffer); 
   void endCommandBuffer(VkCommandBuffer commandBuffer); 

   void createCommandPool(); 
   void allocateCommandBuffers(size_t imageCount); 

   void destroy(); 
}; 
```

In the class, we create the constructor and destructor. We create variables to store the command pool and a vector to store `VkCommandBuffer`. We create one function initially to create the command pool and allocate the command buffers. The next two functions, `beginCommandBuffer` and `endCommandBuffer`, will be called when we want to start and stop recording the command buffer. The `createCommandPool` and `allocateCommandBuffers` functions will be called by `createCommandPoolAndBuffer`.

We will create the `destroy` function to destroy the command buffers when we want the resources to be released to the system. In `CommandBuffer.cpp`, add the necessary includes and the constructor and destructor:

```cpp
#include "DrawCommandBuffer.h" 
#include "VulkanContext.h"

DrawCommandBuffer::DrawCommandBuffer(){} 

DrawCommandBuffer::~DrawCommandBuffer(){}   

```

Then, we add `createCommandPoolAndBuffer`, which takes in the picture count:

```cpp
void DrawCommandBuffer::createCommandPoolAndBuffer(size_t imageCount){ 

   createCommandPool(); 
   allocateCommandBuffers(imageCount); 
}
```

The `createCommandPoolAndBuffer` function will call the `createCommandPool` and `allocateCommandBuffers` functions. First, we create the `createCommandPool` function. Commands have to be sent to a certain queue. We have to specify the queue when we create the command pool:

```cpp
void DrawCommandBuffer::createCommandPool() { 

   QueueFamilyIndices qFamilyIndices = VulkanContext::
                                       getInstance()->getDevice()-> 
                                       getQueueFamiliesIndicesOfCurrentDevice(); 

   VkCommandPoolCreateInfo cpInfo = {}; 

   cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; 
   cpInfo.queueFamilyIndex = qFamilyIndices.graphicsFamily; 
   cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; 

   if (vkCreateCommandPool(VulkanContext::getInstance()->
       getDevice()->logicalDevice, &cpInfo, nullptr, &commandPool) 
       != VK_SUCCESS) { 
          throw std::runtime_error(" failed to create command pool !!"); 
   } 

} 
```

To start, we get the queue family indices for the current device. To create the command pool, we have to populate the `VkCommandPoolCreateInfo` struct. As usual, we specify the type. Then, we set the queue family index in which the pool has to be created. After that, we set the `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` flags, which will reset the values of the command buffer every time. We then use the `vkCreateCommandPool` function by passing in the logical device and the info struct to get the command pool. Next, we create the `allocateCommandBuffers` function:

```cpp
void DrawCommandBuffer::allocateCommandBuffers(size_t imageCount) { 

   commandBuffers.resize(imageCount); 

   VkCommandBufferAllocateInfo cbInfo = {}; 
   cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; 
   cbInfo.commandPool = commandPool; 
   cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; 
   cbInfo.commandBufferCount = (uint32_t)commandBuffers.size(); 

   if (vkAllocateCommandBuffers(VulkanContext::getInstance()->
      getDevice()->logicalDevice, &cbInfo, commandBuffers.data()) 
      != VK_SUCCESS) { 

         throw std::runtime_error(" failed to allocate 
            command buffers !!"); 
   } 

} 

```

We resize the `commandBuffers` vector. Then, to allocate the command buffers, we have to populate `VkCommandBufferAllocateInfo`. We first set the type of the struct and the command pool. Then, we have to specify the level of the command buffers. You can have a chain of command buffers, with the primary command buffer containing the secondary command buffer. For our use, we will set the command buffers as primary. We then set `commandBufferCount`, which is equal to the swapchain pictures.

Then, we allocate the command buffers using the `vkAllocateCommandBuffers` function. We pass in the logical device, the info struct, and the command buffers to allocate memory for the command buffers.

Then, we add `beginCommandBuffer`. This takes in the current command buffer to start recording commands in it:

```cpp

void DrawCommandBuffer::beginCommandBuffer(VkCommandBuffer commandBuffer){ 

   VkCommandBufferBeginInfo cbBeginInfo = {}; 

   cbBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; 

   cbBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; 

   if (vkBeginCommandBuffer(commandBuffer, &cbBeginInfo) != VK_SUCCESS) { 

         throw std::runtime_error(" failed to begin command buffer !!"); 
   } 

}
```

To record command buffers, we also have to populate `VkCommandBufferBeginInfoStruct`. Once again, we specify the struct type and the `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` flag. This enables us to schedule the command buffer for the next frame while the last frame is still in use. `vkBeginCommandBuffer` is called to start recording the commands by passing in the current command buffer.

Next, we add in the `endCommandBuffer` function. This function just calls `vkEndCommandBuffer` to stop recording to the command buffer:

```cpp
void DrawCommandBuffer::endCommandBuffer(VkCommandBuffer commandBuffer){ 

   if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) { 

         throw std::runtime_error(" failed to record command buffer"); 
   } 

} 
```

We can then destroy the command buffers and the pool using the `Destroy` function. Here, we just destroy the pool, which will destroy the command buffers as well:

```cpp
void DrawCommandBuffer::destroy(){ 

   vkDestroyCommandPool(VulkanContext::getInstance()->
      getDevice()->logicalDevice, commandPool, nullptr); 

} 
```

In the `VulkanApplication.h` file, include `DrawCommandBuffer.h` and create an object of this class. In `VulkanApplication.cpp`, in the `VulkanInit` function, after creating `renderViewsAndFrameBuffers`, create `DrawCommandBuffer`:

```cpp
   renderTexture = new RenderTexture(); 
   renderTexture->createViewsAndFramebuffer(swapChain->swapChainImages, 
         swapChain->swapChainImageFormat, 
         swapChain->swapChainImageExtent, 
         renderPass->renderPass); 

   drawComBuffer = new DrawCommandBuffer(); 
   drawComBuffer->createCommandPoolAndBuffer(swapChain->
     swapChainImages.size());
```

# Beginning and ending Renderpass

Along with the commands being recorded in each frame, the renderpass is also processed for each frame where the color and the depth information is reset. So, since we only have color attachments in each frame, we have to clear the color information for each frame as well. Go back to the `Renderpass.h` file and add two new functions, called `beginRenderPass` and `endRenderPass`, in the class, as follows:

```cpp
class Renderpass 
{ 
public: 
   Renderpass(); 
   ~Renderpass(); 

   VkRenderPass renderPass; 

   void createRenderPass(VkFormat swapChainImageFormat); 

   void beginRenderPass(std::array<VkClearValue, 1> 
      clearValues, VkCommandBuffer commandBuffer, VkFramebuffer 
      swapChainFrameBuffer, VkExtent2D swapChainImageExtent); 

   void endRenderPass(VkCommandBuffer commandBuffer); 

   void destroy(); 
}; 

```

In `RenderPass.cpp`, add the implementation of the `beginRenderPass` function:

```cpp
void Renderpass::beginRenderPass(std::array<VkClearValue, 1> clearValues, 
    VkCommandBuffer commandBuffer, VkFramebuffer swapChainFrameBuffer, 
    VkExtent2D swapChainImageExtent) { 

   VkRenderPassBeginInfo rpBeginInfo = {}; 
   rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; 
   rpBeginInfo.renderPass = renderPass; 
   rpBeginInfo.framebuffer = swapChainFrameBuffer; 
   rpBeginInfo.renderArea.offset = { 0,0 }; 
   rpBeginInfo.renderArea.extent = swapChainImageExtent; 

   rpBeginInfo.pClearValues = clearValues.data(); 
   rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size()); 

   vkCmdBeginRenderPass(commandBuffer,&rpBeginInfo,
      VK_SUBPASS_CONTENTS_INLINE); 
} 
```

We then populate the `VkRenderPassBeginInfo` struct. In this, we specify the struct type, pass in the renderpass and the current framebuffer, set the render area as the whole viewport, and pass in the clear value and the count. The clear value is the color value we want to clear the screen with, and the count would be `1`, as we would like to clear only the color attachment.

To begin the renderpass, we pass in the current command buffer, the info struct, and specify the third parameter as `VK_SUBPASS_CONTENTS_INLINE`, specifying that the renderpass commands are bound to the primary command buffer.

In the `endCommandBuffer` function, we finish the `Renderpass` for the current frame:

```cpp
void Renderpass::endRenderPass(VkCommandBuffer commandBuffer){ 

   vkCmdEndRenderPass(commandBuffer); 
} 

```

To end the `Renderpass`, the `vkCmdEndRenderPass` function is called and the current command buffer is passed in.

We have the required classes to get the clear screen going. Now, let's go to the Vulkan Application class and add some lines of code to get it working.

# Creating the clear screen

In the `VulkanApplication.h` file, we will add three new functions, called `drawBegin`, `drawEnd`, and `cleanup`. `drawBegin` will be called before we pass in any draw commands, and `drawEnd` will be called once the drawing is done and the frame is ready to be presented to the viewport. In the `cleanup` function, we will destroy all the resources.

We will also create two variables. The first is `uint32_t`, to get the current picture from the swapchain, and the second is `currentCommandBuffer` of the `VkCommandBuffer` type to get the current command buffer:

```cpp
public: 

   static VulkanApplication* getInstance(); 
   static VulkanApplication* instance; 

   ~VulkanApplication(); 

   void initVulkan(GLFWwindow* window); 

   void drawBegin(); 
   void drawEnd(); 
void cleanup(); 

private: 

   uint32_t imageIndex = 0; 
   VkCommandBuffer currentCommandBuffer; 

   //surface 
   VkSurfaceKHR surface; 

```

In the `VulkanApplication.cpp` file, we add the implementation of the `drawBegin` and `drawEnd` functions:

```cpp
void VulkanApplication::drawBegin(){ 

   vkAcquireNextImageKHR(VulkanContext::getInstance()->
      getDevice()->logicalDevice, 
         swapChain->swapChain, 
         std::numeric_limits<uint64_t>::max(), 
         NULL,  
         VK_NULL_HANDLE, 
         &imageIndex); 

   currentCommandBuffer = drawComBuffer->commandBuffers[imageIndex]; 

   // Begin command buffer recording 
   drawComBuffer->beginCommandBuffer(currentCommandBuffer); 

   // Begin renderpass 
   VkClearValue clearcolor = { 1.0f, 0.0f, 1.0f, 1.0f }; 

   std::array<VkClearValue, 1> clearValues = { clearcolor }; 

   renderPass->beginRenderPass(clearValues, 
         currentCommandBuffer, 
         renderTexture->swapChainFramebuffers[imageIndex], 
         renderTexture->_swapChainImageExtent); 

} 
```

First, we acquire the next picture from the swap chain. This is done using the Vulkan `vkAcquireNextImageKHR` API call. To do this, we pass in the logical device, the swapchain instance. Next, we need to pass in the timeout, for which we pass in the maximum numerical value as we don't care about the time limit. The next two variables are kept as null. These require a semaphore and a fence, which we will discuss in a later chapter. Finally, we pass in the `imageIndex` itself.

Then, we get the current command buffer from the command buffers vector. We begin recording the command buffer by calling `beginCommandBuffer` and the commands will be stored in the `currentCommandBuffer` object. We now start the renderpass. In this, we pass the clear color value, which is the color purple, because *why not?* ! Pass in the current `commandbuffer`, the frame buffer, and the picture extent.

We can now implement the `drawEnd` function:

```cpp
void VulkanApplication::drawEnd(){ 

   // End render pass commands 
   renderPass->endRenderPass(currentCommandBuffer); 

   // End command buffer recording 
   drawComBuffer->endCommandBuffer(currentCommandBuffer); 

   // submit command buffer 
   VkSubmitInfo submitInfo = {}; 
   submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
   submitInfo.commandBufferCount = 1; 
   submitInfo.pCommandBuffers = &currentCommandBuffer; 

   vkQueueSubmit(VulkanContext::getInstance()->getDevice()->
      graphicsQueue, 1, &submitInfo, NULL); 

   // Present frame 
   VkPresentInfoKHR presentInfo = {}; 
   presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR; 
   presentInfo.swapchainCount = 1; 
   presentInfo.pSwapchains = &swapChain->swapChain; 
   presentInfo.pImageIndices = &imageIndex; 

   vkQueuePresentKHR(VulkanContext::getInstance()->
       getDevice()->presentQueue, &presentInfo); 

   vkQueueWaitIdle(VulkanContext::getInstance()->
       getDevice()->presentQueue); 

} 
```

We end the renderpass and stop recording to the command buffer. Then, we have to submit the command buffer and present the frame. To submit the command buffer, we create a `VkSubmitInfo` struct and populate it with the struct type, the buffer count, which is 1 per frame, and the command buffer itself. The command is submitted to the graphics queue by calling `vkQueueSubmit` and passing in the graphics queue, the submission count, and the submit info.

Once the frame is rendered, it is presented to the viewport using the present queue.

To present the scene once it is drawn, we have to create and populate the `VkPresentInfoKHR` struct. For presentation, the picture is sent back to the swapchain. When we create the info and set the type of the struct, we also have to set the swapchain, the image index, and the swapchain count, which is 1.

We then present the picture using `vkQueuePresentKHR` by passing in the present queue and the present info to the function. At the end, we wait for the host to finish the presentation operation of a given queue using the `vkQueueWaitIdle` function, which takes in the present queue. Also, it is better to clean up the resources when you are done with them, so add the `cleanup` function as well:

```cpp
void VulkanApplication::cleanup() { 

   vkDeviceWaitIdle(VulkanContext::getInstance()->
     getDevice()->logicalDevice); 

   drawComBuffer->destroy(); 
   renderTexture->destroy(); 
   renderPass->destroy(); 
   swapChain->destroy(); 

   VulkanContext::getInstance()->getDevice()->destroy(); 

   valLayersAndExt->destroy(vInstance->vkInstance, 
      isValidationLayersEnabled); 

   vkDestroySurfaceKHR(vInstance->vkInstance, surface, nullptr);   
   vkDestroyInstance(vInstance->vkInstance, nullptr); 

} 
 delete drawComBuffer;
 delete renderTarget;
 delete renderPass;
 delete swapChain;
 delete device;

 delete valLayersAndExt;
 delete vInstance;

 if (instance) {
  delete instance;
  instance = nullptr;
 }
```

When we are destroying the objects, we have to call `vkDeviceWaitIdle` to stop using the device. Then, we destroy the objects in reverse order. So, we destroy the command buffer first, then the render texture resources, then renderpass, and then the swapchain. We then destroy the device, validation layer, surface, and finally, the Vulkan instance. Finally, we also delete the class instances we created for `DrawCommandBuffer`, `RenderTarget`, `Renderpass`, `Swapchain`, `Device`, `ValidationLayersAndExtensions`, and `VulkanInstance`.

And finally, we also delete the instance of `VulkanContext` as well and set it to `nullptr` after deleting it.

In the `source.cpp` file, in the `while` loop, call the `drawBegin` and `drawEnd` functions. Then call the `cleanup` function after the loop:

```cpp
#define GLFW_INCLUDE_VULKAN 
#include<GLFW/glfw3.h> 

#include "VulkanApplication.h" 

int main() { 

   glfwInit(); 

   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); 
   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); 

   GLFWwindow* window = glfwCreateWindow(1280, 720, 
                        "HELLO VULKAN ", nullptr, nullptr); 

   VulkanApplication::getInstance()->initVulkan(window); 

   while (!glfwWindowShouldClose(window)) { 

         VulkanApplication::getInstance()->drawBegin(); 

         // draw command  

         VulkanApplication::getInstance()->drawEnd(); 

         glfwPollEvents(); 
   }               

   VulkanApplication::getInstance()->cleanup(); 

   glfwDestroyWindow(window); 
   glfwTerminate(); 

   return 0; 
} 

```

You will see a purple viewport, as follows, when you build and run the command:

![](img/e6152ccd-b527-4376-b15f-b536ea987712.png)

The screen looks OK, but if you look at the console, you will see the following error, which says that when we call `vkAcquireNextImageKHR`, the semaphore and fence cannot both be `NULL`:

![](img/e720f47c-eff8-4fde-9e32-8689885d1073.png)

# Summary

In this chapter, we looked at the creation of the swapchain, renderpass, render views, framebuffers, and the command buffers. We also looked at what each does and why they are important for rendering a clear screen.

In the next chapter, we will create the resources that will enable us to render geometry to the viewport. Once we have the object resources ready, we will render the objects. We will then explore semaphores and fences and why they are needed.