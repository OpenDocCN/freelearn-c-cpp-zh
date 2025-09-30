# Getting Started with Vulkan

In the previous three chapters, we did our rendering using OpenGL. Although OpenGL is good for developing prototypes and getting your rendering going faster, it does have its weaknesses. For one, OpenGL is very driver-dependent, which makes it slower and less predictable when it comes to performance, which is why we prefer Vulkan for rendering.

In this chapter, we will cover the following topics:

*   About Vulkan
*   Configuring Visual Studio
*   Vulkan validation layers and extensions
*   Vulkan instances
*   The Vulkan Context class
*   Creating the window surface
*   Picking a physical device and creating a logical device

# About Vulkan

With OpenGL, developers have to depend on vendors such as NVIDIA, AMD, and Intel to release appropriate drivers so that they can increase the performance of their games before they are released. This is only possible if the developer is working closely with the vendor. If not, the vendor will only be able to release optimized drivers after the release of the game, and it could take a couple of days to release the new drivers.

Furthermore, if you want to port your PC game to a mobile platform and you are using OpenGL as your renderer, you will need to port the renderer to OpenGLES, which is a subset of OpenGL, where the ES stands for Embedded Systems. Although there are a lot of similarities between OpenGL and OpenGLES, there is still additional work to be done to get it to work on other platforms. To alleviate these issues, Vulkan was introduced. Vulkan gives the developer more control by reducing driver impact and providing explicit developer control to make the game perform better.

Vulkan has been developed from the ground up and therefore is not backward compatible with OpenGL. When using Vulkan, you have complete access to the GPU.

With complete GPU access, you also have complete responsibility for implementing the rendering API. Consequently, the downside of using Vulkan is that you have to specify everything when you're developing with it.

All in all, this makes Vulkan a very verbose API where you have to specify everything. However, this also makes it easy to create extensions of the API specifications for Vulkan when GPUs add newer features.

# Configuring Visual Studio

Vulkan is just a rendering API, so we need to create a window and do math. For both, we will use GLFW and GLM, like we did when we created an OpenGL project. To do this, follow these steps:

1.  Create a new Visual Studio C++ project and call it `VulkanProject`.
2.  Copy the `GLFW` and `GLM` folders from the OpenGL project and place them inside the `VulkanProject` folder, under a folder named `Dependencies`.
3.  Download the Vulkan SDK. Go to [https://vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home) and download the Windows version of the SDK, as shown in the following screenshot:

![](img/b4e84ef1-92b7-4289-99da-003f9c0499aa.png)

4.  Install the SDK, as shown in the following screenshot:

![](img/e0b1f20e-59ef-462b-9a8f-c16a4e217576.png)

5.  In the `Dependencies` directory, create a new folder called `Vulkan`. Copy and paste the `Lib` and include the folder from the Vulkan SDK folder in `C:\ drive`, as shown in the following screenshot:

![](img/734566c4-d064-467a-abfd-0afc5c4c1435.png)

6.  In the Visual Studio project, create a new blank `source.cpp` file. Open up the Vulkan Project properties and add the `include` directory to C/C+ | General | Additional Include Directory.

7.  Make sure that All Configurations and All Platforms are selected in the Configuration and Platform drop-down lists, as shown in the following screenshot:

![](img/405981b3-65ce-41a2-9cf2-d8a11977c6f4.png)

8.  Add the Library Directories under the Linker | General section, as shown in the following screenshot:

![](img/4a23c330-5cbd-4ec9-bb32-379484c85e9b.png)

9.  In Linker | Input, set the libraries that you want to use, as shown in the following screenshot:

![](img/3de238cf-8de9-4f3b-a2d6-b4b1d761aff5.png)

With this prep work out of the way, let's check whether our window creation works properly:

1.  In `source.cpp`, add the following code:

```cpp

#defineGLFW_INCLUDE_VULKAN 
#include<GLFW/glfw3.h> 

int main() { 

   glfwInit(); 

   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); 
   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); 

   GLFWwindow* window = glfwCreateWindow(1280, 720, "HELLO VULKAN ", nullptr, nullptr); 

   while (!glfwWindowShouldClose(window)) { 

         glfwPollEvents(); 
   } 

   glfwDestroyWindow(window); 
   glfwTerminate(); 

   return 0; 
}
```

First, we include `glfw3.h` and ask `GLFW` to include some Vulkan-related headers. Then, in the main function, we initialize `GLFW` by calling `glfwInit()`. Then, we call the `glfwWindowHint` functions. The first `glfwWindowHint` function doesn't create the OpenGL context since it is created by `Glfw` by default. In the next function, we disable resizing for the window we are about to create.

Then, we create the 1,280 x 720 window in a similar way to when we created the window in the OpenGL project. We create a `while` loop that checks whether the window should be closed. If the window doesn't need to be closed, we will poll the system events. Once this is done, we will destroy the window, terminate `glfw`, and return `0`.

2.  This should give us a window to work with. Run the application in debug mode as an x64 executable to see the window being displayed and saying HELLO VULKAN, as shown in the following screenshot:

![](img/f8b4a3d5-0065-4d80-91a3-990828b2e507.png)

# Vulkan validation layers and extensions

Before we jump into creating the Vulkan application, we have to check for application validation layers and extensions. Let's go over these in more detail:

*   **Validation layers**: Since so much control is given to developers, it is also possible for the developers to implement the Vulkan applications in an incorrect manner. The Vulkan validation layers check for such errors and tell the developer that they are doing something wrong and need to fix it.
*   **Extensions**: Over the course of the development of the Vulkan API, new features may be introduced to newer GPUs. To keep Vulkan up to date, we need to extend its functionality by adding extensions.

One example of this is the introduction of Ray Tracing in the RTX series of GPUs. In Vulkan, a new extension was created to support this change in the hardware by NVIDIA, that is, `Vk_NV_ray_tracing`. If our game uses this extension, we can check whether the hardware supports it.

Similar extensions can be added and checked at the application level as well. One such extension is the Debug report extension, which we can generate if something goes wrong when we're implementing Vulkan. Our first class will add this functionality to the application to check for application validation layers and extensions.

Let's start creating our first class. Create a new class called `AppValidationLayersAndExtensions`. In `AppValidationLayersAndExtensions.h`, add the following code:

```cpp
#pragmaonce 

#include<vulkan\vulkan.h> 
#include<vector> 
#include<iostream> 

#defineGLFW_INCLUDE_VULKAN 
#include<GLFW\glfw3.h> 

classAppValidationLayersAndExtensions { 

public: 
   AppValidationLayersAndExtensions(); 
   ~AppValidationLayersAndExtensions(); 

   const std::vector<constchar*> requiredValidationLayers = { 
         "VK_LAYER_LUNARG_standard_validation" 
   }; 

   bool checkValidationLayerSupport(); 
   std::vector<constchar*>getRequiredExtensions
     (boolisValidationLayersEnabled); 

   // Debug Callback 
   VkDebugReportCallbackEXT callback; 

   void setupDebugCallback(boolisValidationLayersEnabled, 
      VkInstancevkInstance); 
   void destroy(VkInstanceinstance, boolisValidationLayersEnabled); 

   // Callback 

* pCreateInfo, VkResultcreateDebugReportCallbackEXT( 
       VkInstanceinstance, 
       constVkDebugReportCallbackCreateInfoEXT         
       constVkAllocationCallbacks* pAllocator, 
       VkDebugReportCallbackEXT* pCallback) { 

         auto func = (PFN_vkCreateDebugReportCallbackEXT)
                     vkGetInstanceProcAddr(instance, 
                     "vkCreateDebugReportCallbackEXT"); 

         if (func != nullptr) { 
               return func(instance, pCreateInfo, pAllocator, pCallback); 
         } 
         else { 
               returnVK_ERROR_EXTENSION_NOT_PRESENT; 
         } 

   } 

   void DestroyDebugReportCallbackEXT( 
         VkInstanceinstance, 
         VkDebugReportCallbackEXTcallback, 
         constVkAllocationCallbacks* pAllocator) { 

         auto func = (PFN_vkDestroyDebugReportCallbackEXT)
                     vkGetInstanceProcAddr(instance, 
                     "vkDestroyDebugReportCallbackEXT"); 
         if (func != nullptr) { 
               func(instance, callback, pAllocator); 
         } 
   } 

}; 
```

We include `vulkan.h`, `iostream`, `vector`, and `glfw`. Then, we create a vector called `requiredValidationLayers`; this is where we pass `VK_LAYER_LUNARG _standard_validation`. For our application, we will need the standard validation layer, which has all the validation layers in it. If we only need specific validation layers, then we can specify them individually as well. Next, we create two functions: one for checking the support for validation layers and one for getting the required extensions.

To generate a report in case an error occurs, we need a debug callback. We will add two functions to it: one to set up the debug callback and one to destroy it. These functions will call the `debug`, `create`, and `destroy` functions; they will call `vkGetInstanceProcAddr` to get the pointers for the `vkCreateDebugReportCallbackEXT` and `vkDestroyDebugReportCallbackEXT` pointer functions so that we can call them.

It would be better if it were less confusing to generate a debug report, but unfortunately, this is how it must be done. However, we only have to do this once. Let's move on to implementing `AppValidationLayersAndExtentions.cpp`:

1.  First, we add the constructor and destructor, as follows:

```cpp
AppValidationLayersAndExtensions::AppValidationLayersAndExtensions(){} 

AppValidationLayersAndExtensions::~AppValidationLayersAndExtensions(){} 
Then we add the implementation to checkValidationLayerSupport(). 

bool AppValidationLayersAndExtensions::checkValidationLayerSupport() { 

   uint32_t layerCount; 

   // Get count of validation layers available 
   vkEnumerateInstanceLayerProperties(&layerCount, nullptr); 

   // Get the available validation layers names  
   std::vector<VkLayerProperties>availableLayers(layerCount); 
   vkEnumerateInstanceLayerProperties(&layerCount,
   availableLayers.data()); 

   for (const char* layerName : requiredValidationLayers) { //layers we
   require 

         // boolean to check if the layer was found 
         bool layerFound = false; 

         for (const auto& layerproperties : availableLayers) { 

               // If layer is found set the layar found boolean to true 
               if (strcmp(layerName, layerproperties.layerName) == 0) { 
                     layerFound = true; 
                     break; 
               } 
         } 

         if (!layerFound) { 
               return false; 
         } 

         return true; 

   } 

}
```

To check the supported validation layers, call the `vkEnumerateInstanceLayerProperties` function twice. We call it the first time to get the number of validation layers that are available. Once we have the count, we call it again to populate it with the names of the layers.

We create an `int` called `layerCount` and pass it in the first time we call `vkEnumerateInstanceLayerProperties`. The function takes two parameters: the first is the count and the second is initially kept `null`. Once the function is called, we will know how many validation layers are available. For the names of the layers, we create a new vector called `availableLayers` of the `VkLayerProperties` type and initialize it with `layerCount`. Then, the function is called again, and this time we pass in `layerCount` and the vector as parameters to store the information. After, we make a check between the required layers and the available layers. If the validation layer was found, the function will return `true`. If not, it will return `false`.

2.  Next, we need to add the `getRequiredInstanceExtentions` function, as follows:

```cpp
std::vector<constchar*>AppValidationLayersAndExtensions::getRequiredExtensions(boolisValidationLayersEnabled) { 

   uint32_t glfwExtensionCount = 0; 
   constchar** glfwExtensions; 

   // Get extensions 
   glfwExtensions = glfwGetRequiredInstanceExtensions
                    (&glfwExtensionCount); 

   std::vector<constchar*>extensions(glfwExtensions, glfwExtensions 
     + glfwExtensionCount); 

   //debug report extention is added. 

   if (isValidationLayersEnabled) { 
         extensions.push_back("VK_EXT_debug_report"); 
   } 

   return extensions; 
}
```

The `getRequiredInstanceExtensions` phrase will get all the extensions that are supported by `GLFW`. It takes a Boolean to check whether the validation layers are enabled and returns a vector with the names of the supported extensions. In this function, we create a `unint32_t` called `glfwExtensionCount` and a `const` char for storing the names of the extensions. We call `glfwGetRequiredExtentions`, pass in `glfwExtentionCount`, and set it so that it's equal to `glfwExtensions`. This will store all the required extensions in `glfwExtensions`.

We create a new extensions vector and store the `glfwExtention` names. If we have enabled the validation layer, then we can add an additional extension layer called `VK_EXT_debug_report`, which is the extension for generating a debug report. This extension vector is returned at the end of the function.

3.  Then, we add the debug report callback function, which will generate a report message whenever there is an error, as follows:

```cpp
 staticVKAPI_ATTRVkBool32VKAPI_CALL debugCallback( 
   VkDebugReportFlagsEXTflags, 
   VkDebugReportObjectTypeEXTobjExt, 
   uint64_tobj, 
   size_tlocation, 
   int32_tcode, 
   constchar* layerPrefix, 
   constchar* msg, 
   void* userData) { 

   std::cerr <<"validation layer: "<<msg<< std::endl << std::endl; 

   returnfalse; 

} 
```

4.  Next, we need to create the `setupDebugCallback` function, which will call the `createDebugReportCallbackExt` function, as follows:

```cpp
voidAppValidationLayersAndExtensions::setupDebugCallback(boolisValidationLayersEnabled, VkInstancevkInstance) { 

   if (!isValidationLayersEnabled) { 
         return; 
   } 

   printf("setup call back \n"); 

   VkDebugReportCallbackCreateInfoEXT info = {}; 

   info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT
                _CALLBACK_CREATE_INFO_EXT; 
   info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | 
                VK_DEBUG_REPORT_WARNING_BIT_EXT; 
   info.pfnCallback = debugCallback; // callback function 

   if (createDebugReportCallbackEXT(vkInstance, &info, nullptr, 
     &callback) != VK_SUCCESS) { 

         throw std::runtime_error("failed to set debug callback!"); 
   } 

} 

```

This function takes a Boolean, which will check that the validation layer is enabled. It also takes a Vulkan instance, which we will create after this class.

When creating a Vulkan object, we usually have to populate a struct with the required parameters. So, to create `DebugReportCallback`, we have to populate the `VkDebugReportCallbackCreateInfoExt` struct first. In the struct, we pass in `sType`, which specifies the structure type. We also pass in any flags for error and warning reporting. Finally, we pass in the `callback` function itself. Then, we call the `createDebugReportCallbackExt` function and pass in the instance, the struct, a null pointer for memory allocation, and the `callback` function.  Even though we pass in a null pointer for memory allocation, Vulkan will take care of memory allocation by itself. This function is available if you have a memory allocation function of your own.

5.  Now, let's create the `destroy` function so that we can destroy the debug report `callback` function, as follows:

```cpp
voidAppValidationLayersAndExtensions::destroy(VkInstanceinstance, boolisValidationLayersEnabled){ 

   if (isValidationLayersEnabled) { 
         DestroyDebugReportCallbackEXT(instance, callback, nullptr); 
   } 

} 
```

# Vulkan instances

To use the `AppValidationLayerAndExtension` class, we have to create a Vulkan instance. To do so, follow these steps:

1.  We will create another class called `VulkanInstance`. In `VulkanInstance.h`, add the following code:

```cpp
#pragmaonce 
#include<vulkan\vulkan.h> 

#include"AppValidationLayersAndExtensions.h" 

classVulkanInstance 
{ 
public: 
   VulkanInstance(); 
   ~VulkanInstance(); 

   VkInstance vkInstance; 

   void createAppAndVkInstance(,boolenableValidationLayers  
        AppValidationLayersAndExtensions *valLayersAndExtentions); 

};  
```

We're including `vulkan.h` and `AppValidationLayersAndExtentions.h` since we will need the required validation layers and extensions when we create the Vulkan instance. We add the constructor, destructor, and instance of `VkInstance`, as well as a function called `ceeateAppAndVkInstance`. This function takes a Boolean that checks whether the validation layers are enabled, as well as `AppValidationLayersAndExtensions`. That's it for the header.

2.  In the `.cpp` file, add the following code:

```cpp
#include"VulkanInstance.h" 

VulkanInstance::VulkanInstance(){} 

VulkanInstance::~VulkanInstance(){}
```

3.  Then add the `createAppAndVkInstance` function, which will allow us to create the Vulkan instance, as follows:

```cpp
voidVulkanInstance::createAppAndVkInstance(boolenableValidationLayers, AppValidationLayersAndExtensions *valLayersAndExtentions) { 

   // links the application to the Vulkan library 

   VkApplicationInfo appInfo = {}; 
   appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; 
   appInfo.pApplicationName = "Hello Vulkan"; 
   appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); 
   appInfo.pEngineName = "SidTechEngine"; 
   appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); 
   appInfo.apiVersion = VK_API_VERSION_1_0; 

   VkInstanceCreateInfo vkInstanceInfo = {}; 
   vkInstanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; 
   vkInstanceInfo.pApplicationInfo = &appInfo; 

   // specify extensions and validation layers 
   // these are global meaning they are applicable to whole program 
      not just the device 

   auto extensions = valLayersAndExtentions->
                   getRequiredExtensions(enableValidationLayers); 

   vkInstanceInfo.enabledExtensionCount = static_cast<uint32_t>
      (extensions.size());; 
   vkInstanceInfo.ppEnabledExtensionNames = extensions.data(); 

   if (enableValidationLayers) { 
     vkInstanceInfo.enabledLayerCount = static_cast<uint32_t>
     (valLayersAndExtentions->requiredValidationLayers.size()); 
     vkInstanceInfo.ppEnabledLayerNames = 
     valLayersAndExtentions->requiredValidationLayers.data(); 
   } 
   else { 
         vkInstanceInfo.enabledLayerCount = 0; 
   } 
  if (vkCreateInstance(&vkInstanceInfo, nullptr, &vkInstance) !=
   VK_SUCCESS) {
   throw std::runtime_error("failed to create vkInstance ");
  }
}   
```

In the preceding function, we have to populate `VkApplicationInfostruct`, which will be required when we create `VkInstance`. Then, we create the `appInfo` struct. Here, the first parameter we specify is the `struct` type, which is of the `VK_STRUCTURE_TYPE_APPLICATION_INFO` type. The next parameter is the application name itself and is where we specify the application version, which is 1.0\. Then, we specify the engine name and version. Finally, we specify the Vulkan API version to use.

Once the application `struct` has been populated, we can create the `vkInstanceCreateInfo` struct, which will create the Vulkan instance. In the struct instance we created – just like all the structs before this – we have to specify the struct with the `struct` type, which is `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO`.

Then, we have to pass in the application info struct. We have to specify the Vulkan extension and validation layers and counts. This information is retrieved from the `AppValidationLayersAndExtensions` class. The validation layers are only enabled if the class is in debug mode; otherwise, it is not enabled.

Now, we can create the Vulkan instance by calling the `vkCreateInstance` function. This takes three parameters: the create info instance, an allocator, and the instance variable that's used to store the Vulkan instance. For allocation, we specify `nullptr` and let Vulkan take care of memory allocation. If the Vulkan instance wasn't created, a runtime error will be printed to the console to say that the function failed to create the Vulkan instance.

In order to use this `ValidationAndExtensions` class and the Vulkan instance class, we will create a new Singleton class called `VulkanContext`. We're doing this because we'll need access to some of the Vulkan objects in this class when we create our `ObjectRenderer`.

# The Vulkan Context class

The Vulkan Context class will include all the functionality we need so that we can create our Vulkan renderer. In this class, we will create the validation layer, create the Vulkan application and instance, select the GPU we want to use, create the swapchain, create render targets, create the render pass, and add the command buffers so that we can send our draw commands to the GPU.

We will also add two new functions: `drawBegin` and `drawEnd`. In the `drawBegin` function, we will add the functionality for the preparation stages of drawing. The `drawEnd` function will be called after we have drawn an object and prepared it so that it can be presented to the viewport.

Create a new `.h` class and `.cpp` file. In the `.h` file, include the following code:

```cpp
#defineGLFW_INCLUDE_VULKAN 
#include<GLFW\glfw3.h> 

#include<vulkan\vulkan.h> 

#include"AppValidationLayersAndExtensions.h" 
#include"VulkanInstance.h" 
```

Next, we will create a Boolean called `isValidationLayersEnabled`. This will be set to `true` if the application is running in debug mode and `false` if it's running in release mode:

```cpp
#ifdef _DEBUG 
boolconstbool isValidationLayersEnabled = true; 
#else 
constbool isValidationLayersEnabled = false; 
#endif 
```

Next, we create the class itself, as follows:

```cpp
classVulkanContext { 

public: 

staticVulkanContextn* instance;   
staticVulkanContext* getInstance(); 

   ~VulkanContext(); 

   void initVulkan(); 

private: 

   // My Classes 
   AppValidationLayersAndExtensions *valLayersAndExt; 
   VulkanInstance* vInstance; 

};
```

In the `public` section, we create a static instance and the `getInstance` variable and function, which sets and gets the instance of this class. We add the destructor and add an `initVulkan` function, which will be used to initialize the Vulkan context. In the `private` section, we create an instance of the `AppValidationLayersAndExtentions` and `VulkanInstance` classes. In the `VulkanContext.cpp` file, we set the instance variable to `null`, and, in the `getInstance` function, we check whether the instance was created. If it was not created, we create a new instance, return it, and add the destructor:

```cpp
#include"VulkanContext.h" 

VulkanContext* VulkanContext::instance = NULL; 

VulkanContext* VulkanContext::getInstance() { 

   if (!instance) { 
         instance = newVulkanContext(); 
   } 
   return instance; 
} 

VulkanContext::~VulkanContext(){ 
```

Then, we add the functionality for the `initVulkan` function, as follows:

```cpp
voidVulkanContext::initVulkan() { 

   // Validation and Extension Layers 
   valLayersAndExt = newAppValidationLayersAndExtensions(); 

   if (isValidationLayersEnabled && !valLayersAndExt->
     checkValidationLayerSupport()) { 
         throw std::runtime_error("Validation Layers 
           Not Available !"); 
   } 

   // Create App And Vulkan Instance() 
   vInstance = newVulkanInstance(); 
   vInstance->createAppAndVkInstance(isValidationLayersEnabled, 
      valLayersAndExt); 

   // Debug CallBack 
   valLayersAndExt->setupDebugCallback(isValidationLayersEnabled, 
     vInstance->vkInstance); 

}  
```

First, we create a new `AppValidationLayersAndExtensions` instance. Then, we check whether the validation layers are enabled and check whether the validation layers are supported. If `ValidationLayers` is not available, a runtime error is sent out, saying that the validation layers are not available.

If the validation layers are supported, a new instance of the `VulkanInstance` class is created and the `createAppAndVkInstance` function is called, which creates a new `vkInstance`.

Once this is complete, we call the `setupDebugCallBack` function by passing in the Boolean and `vkInstance`. In the `source.cpp` file, include the `VulkanContext.h` file and call `initVulkan` after the window is created, as follows:

```cpp
 #defineGLFW_INCLUDE_VULKAN 
#include<GLFW/glfw3.h> 

#include"VulkanContext.h" 

int main() { 

   glfwInit(); 

   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); 
   glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); 

   GLFWwindow* window = glfwCreateWindow(1280, 720, "HELLO VULKAN ", 
                        nullptr, nullptr); 

   VulkanContext::getInstance()->initVulkan(); 

   while (!glfwWindowShouldClose(window)) { 

         glfwPollEvents(); 
   }               

   glfwDestroyWindow(window); 
   glfwTerminate(); 

   return 0; 
}
```

Hopefully, you won't get any errors in the console window when you build and run the application. If you do get errors, go through each line of code and make sure there are no spelling mistakes:

![](img/5a01dea9-20d8-4215-824d-c92cbb68c200.png)

# Creating the window surface

We need an interface for the window we created for the current platform so that we can present the images we will render. We use the `VKSurfaceKHR` property to get access to the window surface. To store the surface information that the OS supports, we will call the `glfw` function, `glfwCreateWindowSurface`, to create the surface that's supported by the OS.

In `VulkanContext.h`, add a new variable of the `VkSurfaceKHR` type called `surface`, as follows:

```cpp
private: 

   //surface 
   VkSurfaceKHR surface; 

```

Since we need access to the window instance we created in `source.cpp`, change the `initVulkan` function so that it accepts a `GLFWwindow`, as follows:

```cpp
   void initVulkan(GLFWwindow* window); 

```

In `VulkanContext.cpp`, change the `initVulkan` implementation as follows and call the `glfwCreateWindowSurface` function, which takes in the Vulkan instance and the window. Next, pass in `null` for the allocator and the surface to create the surface object:

```cpp
 void VulkanContext::initVulkan(GLFWwindow* window) { 

   // -- Platform Specific 

   // Validation and Extension Layers 
   valLayersAndExt = new AppValidationLayersAndExtensions(); 

   if (isValidationLayersEnabled && !valLayersAndExt->
      checkValidationLayerSupport()) { 
         throw std::runtime_error("Requested Validation Layers
            Not Available !"); 
   } 

   // Create App And Vulkan Instance() 
   vInstance = new VulkanInstance(); 
   vInstance->createAppAndVkInstance(isValidationLayersEnabled, 
     valLayersAndExt); 

   // Debug CallBack 
   valLayersAndExt->setupDebugCallback(isValidationLayersEnabled, 
    vInstance->vkInstance); 

   // Create Surface 
   if (glfwCreateWindowSurface(vInstance->vkInstance, window, 
      nullptr, &surface) != VK_SUCCESS) { 

         throw std::runtime_error(" failed to create window 
           surface !"); 
   } 
} 

```

Finally, in `source.cpp`, change the `initVulkan` function, as follows:

```cpp
   GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, 
                        "HELLO VULKAN ", nullptr, nullptr); 

   VulkanContext::getInstance()->initVulkan(window); 
```

# Picking a physical device and creating a logical device

Now, we will create the `Device` class, which will be used to go through the different physical devices we have. We will choose one to render our application. To check whether your GPU is compatible with Vulkan, check the compatibility list on your GPU vendor's site or go to [https://en.wikipedia.org/wiki/Vulkan_(API)](https://en.wikipedia.org/wiki/Vulkan_(API)).

Basically, any NVIDIA GPU from the Geforce 600 series and AMD GPU from the Radeon HD 2000 series and later should be supported. To access the physical device and create a logical device, we will create a new class that will allow us to access it whenever we want. Create a new class called `Device`. In `Device.h`, add the following includes:

```cpp
#include<vulkan\vulkan.h> 
#include<stdexcept> 

#include<iostream> 
#include<vector> 
#include<set> 

#include"VulkanInstance.h" 
#include"AppValidationLayersAndExtensions.h" 
```

We will also add a couple of structs for the sake of convenience. The first is called `SwapChainSupportDetails`; it has access to `VkSurfaceCapabilitiesKHR`, which contains all the required details about the surface. We'll also add the `surfaceFormats` vector of the `VkSurfaceFormatKHR` type, which keeps track of all the different image formats the surface supports, and the `presentModes` vector of the `VkPresentModeKHR` type, which stores the presentation modes that the GPU supports.

Rendered images will be sent to the window surface and displayed. This is how we are able to see the final rendered image using a renderer, such as OpenGL or Vulkan. Now, we can show these images to the window one at a time, which is fine if we want to look at a still image forever. However, when we run a game that is updated every 16 milliseconds (60 times in a second), there may be cases where the image has not been fully rendered, but it would be time to display it. At this point, we will see half-rendered images, which leads to screen tearing.

To avoid this, we use double buffering. This allows us to render the image so that it has two different images, known as the front buffer and the back buffer, and ping-pong between them. Then, we present the buffer that has finished rendering and display it to the viewport while the next frame is still being rendered, as shown in the following diagram. There are different ways to present the image as well. We will look at these different presentation modes when we create the swapchain:

![](img/ee9c08d1-1be2-4404-8aca-76b1e698d73a.png)

We need to create a struct to track the surface properties, format, and presentation modes, as follows:

```cpp
structSwapChainSupportDetails { 

   VkSurfaceCapabilitiesKHR surfaceCapabilities; // size and images 
                                                  in swapchain 
   std::vector<VkSurfaceFormatKHR> surfaceFormats; 
   std::vector<VkPresentModeKHR> presentModes; 
}; 
```

A GPU also has what is called `QueueFamilies`. Commands are sent to the GPU and then executed using queues. There are separate queues for different kinds of work. Render commands are sent to render queues, compute commands are sent to compute queues, and there are also presentation queues for presenting images. We also need to know which queues the GPU supports and how many of the queues are present.

The renderer, compute, and presentation queues can be combined and are known as queue families. These queues can be combined in different ways to form a number of queue families. This means that there can be a combination of render and presentation queues that forms one queue family, while another family may just contain compute queues. Therefore, we have to check whether we have at least one queue family with graphics and presentation queues. This is because we need a graphics queue to pass our rendering commands into and a presentation queue to present the image after we render it.

We will add one more struct to check for both, as follows:

```cpp
structQueueFamilyIndices { 

   int graphicsFamily = -1; 
   int presentFamily = -1; 

   bool arePresent() { 
         return graphicsFamily >= 0 && presentFamily >= 0; 
   } 
}; 
```

Now, we will create the `Device` class itself. After creating the class, we add the constructor and destructor, as follows:

```cpp
 { 

public: 

   Device(); 
   ~Device();  
```

Then, we need to add some variables so that we can store the physical device, the `SwapChainSupportDetails`, and the `QueueFamilyIndices`, as follows:

```cpp
   VkPhysicalDevice physicalDevice; 
   SwapChainSupportDetails swapchainSupport; 
   QueueFamilyIndices queueFamiliyIndices; 

```

To create double buffering, we have to check that the device supports it. This is done using the `VK_KHR_SWAPCHAIN_EXTENSION_NAME` extension, which checks for a swapchain. First, we create a vector of the `char*` const and pass in the extension name, as follows:

```cpp
std::vector<constchar*>deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
```

Then, we add the `pickPhysicalDevice` function, which will be selected depending on whether the device is suitable. While checking for suitability, we will check whether the selected device supports the swapchain extension, get the swapchain support details, and get the queue family indices, as follows:

```cpp
   void pickPhysicalDevice (VulkanInstance* vInstance, 
     VkSurfaceKHR surface); 

   bool isDeviceSuitable(VkPhysicalDevice device, 
     VkSurfaceKHR surface); 

   bool checkDeviceExtensionSupported(VkPhysicalDevice device) ; 
   SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice 
      device, VkSurfaceKHR surface); 
   QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, 
      VkSurfaceKHR surface); 
```

We will also add a getter function to get the queue families of the current device, as follows:

```cpp
 QueueFamilyIndicesgetQueueFamiliesIndicesOfCurrentDevice();  
```

Once we have the physical device we want to use, we will create an instance of the logical device. The logical device is an interface for the physical device itself. We will use the logical device to create buffers and so on. We will also store the current device graphics and present a queue so that we can send the graphics and presentation commands. Finally, we will add a `destroy` function, which is used to destroy the physical and logical devices we created, as follows:

```cpp
   // ++++++++++++++ 
   // Logical device 
   // ++++++++++++++ 

   void createLogicalDevice(VkSurfaceKHRsurface, 
      boolisValidationLayersEnabled, AppValidationLayersAndExtensions 
      *appValLayersAndExtentions); 

   VkDevice logicalDevice; 

   // handle to the graphics queue from the queue families of the gpu 
   VkQueue graphicsQueue; // we can also have seperate queue for 
                            compute, memory transfer, etc. 
   VkQueue presentQueue; // queue for displaying the framebuffer 

   void destroy(); 
}; // End of Device class
```

That's all for the `Device.h` file. Let's move on to `Device.cpp`. First, we include `Device.h` and add the constructor and the destructor, as follows:

```cpp
#include"Device.h" 

Device::Device(){} 

Device::~Device(){ 

} 
```

Now, the real work begins. We need to create the `pickPhysicalDevice` function, which takes a Vulkan instance and the `VkSurface`, as follows:

```cpp

voidDevice::pickPhysicalDevice(VulkanInstance* vInstance, VkSurfaceKHRsurface) { 

   uint32_t deviceCount = 0; 

   vkEnumeratePhysicalDevices(vInstance->vkInstance, &deviceCount, 
      nullptr); 

   if (deviceCount == 0) { 
         throw std::runtime_error("failed to find GPUs with vulkan 
           support !"); 
   } 

   std::cout <<"Device Count: "<< deviceCount << std::endl; 

   std::vector<VkPhysicalDevice>devices(deviceCount); 
   vkEnumeratePhysicalDevices(vInstance->vkInstance, &deviceCount, 
      devices.data()); 

   std::cout << std::endl; 
   std::cout <<"DEVICE PROPERTIES"<< std::endl; 
   std::cout <<"================="<< std::endl; 

   for (constauto& device : devices) { 

         VkPhysicalDeviceProperties  deviceProperties; 

         vkGetPhysicalDeviceProperties(device, &deviceProperties); 

         std::cout << std::endl; 
         std::cout <<"Device name: "<< deviceProperties.deviceName 
                   << std::endl; 

         if (isDeviceSuitable(device, surface)) 
               physicalDevice = device; 

   break; 

   } 

   if (physicalDevice == VK_NULL_HANDLE) { 
         throw std::runtime_error("failed to find suitable GPU !"); 
   } 

} 
```

Here, we are creating an `int32` to store the count of the number of physical devices. We get the number of available GPUs using `vkEnumeratePhysicalDevices` and pass the Vulkan instance, the count, and `null` for the third parameter. This will retrieve the number of available devices. If `deviceCount` is zero, this means that there are no GPUs available. Then, we print the available number of devices to the console.

To get the physical devices themselves, we create a vector called `devices`, which will store the `VkPhysicalDevice` data type; this will store the devices for us. We will call the `vkEnumeratePhysicalDevices` function again, but this time – apart from passing in the Vulkan instance and the device count – we will also store the device information in the vector that we passed in as the third parameter. Then, we will print out the number of devices with the `DEVICE PROPERTIES` heading.

To get the properties of the available devices, we will go through the number of devices and get their properties using `vkGetPhysicalDeviceProperties` before storing them in the variable of the `VkPhysicalDeviceProperties` type.

Now, we need to print out the name of the device and call `DeviceSuitable` on the device. If the device is suitable, we will store it as a `physicalDevice` and break out of the loop. Note that we set the first available device as the device we will be using.

If there is no suitable device, we throw a runtime error to say that a suitable device wasn't found. Let's take a look at the `DeviceSuitable` function:

```cpp
bool Device::isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR 
   surface)  { 

   // find queue families the device supports 

   QueueFamilyIndices qFamilyIndices = findQueueFamilies(device, 
                                       surface); 

   // Check device extentions supported 
   bool extensionSupported = checkDeviceExtensionSupported(device); 

   bool swapChainAdequate = false; 

   // If swapchain extension is present  
   // Check surface formats and presentation modes are supported 
   if (extensionSupported) { 

         swapchainSupport = querySwapChainSupport(device, surface); 
         swapChainAdequate = !swapchainSupport.surfaceFormats.empty() 
                             && !swapchainSupport.presentModes.empty(); 

   } 

   VkPhysicalDeviceFeatures supportedFeatures; 
   vkGetPhysicalDeviceFeatures(device, &supportedFeatures); 

   return qFamilyIndices.arePresent() && extensionSupported && 
     swapChainAdequate && supportedFeatures.samplerAnisotropy; 

} 

```

In this function, we get the queue family indices by calling `findQueueFamilies`. Then, we check whether `VK_KHR_SWAPCHAIN_EXTENSION_NAMEextension` is supported. After this, we check for swapchain support on the device. If the surface formats and presentation modes are not empty, `swapChainAdequateboolean` is set to `true`. Finally, we get the physical device features by calling `vkGetPhysicalDeviceFeatures`.

Finally, we return `true` if the queue families are present, the swapchain extension is supported, the swapchain is adequate, and the device supports anisotropic filtering. Anisotropic filtering is a mode that makes the pixels in the distance clearer.

Anisotropic filtering is a mode that, when enabled, helps sharpen textures that are viewed from extreme angles.

In the following example, the image on the right has anisotropic filtering enabled and the image on the left has it disabled. In the image on the right, the white dashed line is still relatively visible further down the road. However, in the image on the left, the dashed line becomes blurry and pixelated. Therefore, anisotropic filtering is required:

![](img/b63e98b4-c702-4646-82ba-fd69536d5b56.png)

(Taken from [https://i.imgur.com/jzCq5sT.jpg](https://i.imgur.com/jzCq5sT.jpg))

Let's look at the three functions we called in the previous function. First, let's check out the `findQueueFamilies` function:

```cpp
QueueFamilyIndicesDevice::findQueueFamilies(VkPhysicalDevicedevice, VkSurfaceKHRsurface) { 

   uint32_t queueFamilyCount = 0; 

   vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, 
      nullptr); 

   std::vector<VkQueueFamilyProperties>queueFamilies(queueFamilyCount); 

   vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, 
      queueFamilies.data()); 

   int i = 0; 

   for (constauto& queueFamily : queueFamilies) { 

         if (queueFamily.queueCount > 0 && queueFamily.queueFlags 
           &VK_QUEUE_GRAPHICS_BIT) { 
               queueFamiliyIndices.graphicsFamily = i; 
         } 

         VkBool32 presentSupport = false; 
         vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, 
           &presentSupport); 

         if (queueFamily.queueCount > 0 && presentSupport) { 
               queueFamiliyIndices.presentFamily = i; 
         } 

         if (queueFamiliyIndices.arePresent()) { 
               break; 
         } 

         i++; 
   } 

   return queueFamiliyIndices; 
}
```

To get the queue family properties, we call the `vkGetPhysicalDeviceQueueFamilyProperties` function; then, in the physical device, we pass an `int`, which we use to store the number of queue families, and the `null` pointer. This will give us the number of queue families that are available.

Next, for the properties themselves, we create a vector of the `VkQueueFamilyProperties` type, called `queueFamilies`, to store the necessary information. Then, we call `vkGetPhysicalDeviceFamilyProperties` and pass in the physical device, the count, and `queueFamilies` itself to populate it with the required data. We create an `int`, `i`, and initialize it to `0`. This will store the index of the graphics and presentation indices.

In the `for` loop, we check whether each of the queue families supports a graphics queue by looking for `VK_QUEUE_GRAPHICS_BIT`. If they do, we set the graphics family index.

Then, we check for presentation support by passing in the index. This will check whether the same family supports presentation as well. If it supports presentation, we set `presentFamily` to that index.

If the queue family supports graphics and presentation, the graphics and presentation index will be the same.

The following screenshot shows the number of queue families by device and the number of queues in each queue family:

![](img/b1fb0547-fff6-4569-aaa7-9d19500f67cc.png)

There are three queue families on my GPU. The first queue family at the 0^(th) index has 16 queues, the second queue family at the 1^(st) index has one queue, and the third queue family at the 2^(nd) index has eight queues.

The `queueFlags` specify the queues in the queue family. The queues that are supported could be for graphics, compute, transfer, or sparse binding.

After this, we check that both the graphics and presentation indices were found, and then we break out of the loop. Finally, we return `queueFamilyIndices`. I am running the project on an Intel Iris Plus Graphics 650\. This integrated intel GPU has one queue family that supports graphics and the presentation queue. Different GPUs have different queue families and each family may support more than one queue type. Next, let's look at the device extension that is supported. We can check this by using the `checkDeviceExtensionSupported` function, which takes in a physical device, as shown in the following code:

```cpp
 boolDevice::checkDeviceExtensionSupported(VkPhysicalDevicedevice){ 

   uint32_t extensionCount; 

   // Get available device extentions count 
   vkEnumerateDeviceExtensionProperties(device, nullptr, 
     &extensionCount, nullptr); 

   // Get available device extentions 
   std::vector<VkExtensionProperties>availableExtensions(extensionCount); 

   vkEnumerateDeviceExtensionProperties(device, nullptr,  
     &extensionCount, availableExtensions.data()); 

   // Populate with required device exentions we need 
   std::set<std::string>requiredExtensions(deviceExtensions.begin(), 
     deviceExtensions.end()); 

   // Check if the required extention is present 
   for (constauto& extension : availableExtensions) { 
         requiredExtensions.erase(extension.extensionName); 
   } 

   // If device has the required device extention then return  
   return requiredExtensions.empty(); 
} 

```

We get the number of extensions that are supported by the device by calling `vkEnumerateDeviceExtensionProperties` and passing in the physical device, the null pointer, an `int` to store the count in it, and `null`. The actual properties are stored inside the `availableExtensions` vector, which stores the `VkExtensionProperties` data type. By calling `vkEnumerateDeviceExtensionProperties` again, we get the device's extension properties.

We populate the `requiredExtensions` vector with the extension we require. Then, we check the available extension vector with the required extensions. If the required extension is found, we remove it from the vector. This means that the device supports the extension and returns the value from the function, as shown in the following code:

![](img/c7f60c53-6246-4e52-a334-8763320c8bcd.png)

The device I am running has 73 available extensions, as shown in the following code. You can set a breakpoint and take a look at the device extension properties to view the supported extension of the device. The third function we will look at is the `querySwapChainSupport` function, which populates the surface capabilities, surface formats, and presentation modes that are available:

```cpp
SwapChainSupportDetailsDevice::querySwapChainSupport
   (VkPhysicalDevicedevice, VkSurfaceKHRsurface) { 

   SwapChainSupportDetails details; 

   vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, 
      &details.surfaceCapabilities); 

   uint32_t formatCount; 
   vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, 
      nullptr); 

   if (formatCount != 0) { 
         details.surfaceFormats.resize(formatCount); 
         vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, 
            &formatCount, details.surfaceFormats.data()); 
   } 

   uint32_t presentModeCount; 
   vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, 
     &presentModeCount, nullptr); 

   if (presentModeCount != 0) { 

         details.presentModes.resize(presentModeCount); 
         vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, 
           &presentModeCount, details.presentModes.data()); 
   } 

   return details; 
} 
```

To get the surface capabilities, we call `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` and pass in the device, that is, `surface`, to get the surface capabilities. To get the surface format and presentation modes, we call `vkGetPhysicalDeviceSurfaceFormatKHR` and `vkGetPhysicalDeviceSurfacePresentModeKHR` twice.

The first time we call the `vkGetPhysicalDeviceSurfacePresentModeKHR` function, we get the number of formats and modes that are present; we call it a second time to get the formats and the modes that have been populated and stored in the vectors of the struct.

Here are the capabilities of my device surface:

![](img/c5c9f1c3-30f8-4dea-8b8f-cc6facc73daf.png)

So, the minimum image count is two, meaning that we can add double buffering. These are the surface formats and the color space that my device supports:

![](img/8887e334-8668-4c48-9e42-8831c766f2b0.png)

Here are the presentation modes that are supported by my device:

![](img/76276022-dd84-42d6-a807-7110aa899c6e.png)

So, it seems that my device only supports the immediate mode. We will see the use of this in the ahead chapters. After getting the physical device properties, we set the getter function for the `queueFamiliyIndices`, as follows:

```cpp
QueueFamilyIndicesDevice::getQueueFamiliesIndicesOfCurrentDevice() { 

   return queueFamiliyIndices; 
} 
```

Now, we can create the logical device by using the `createLogicalDevice` function.

To create the logical device, we have to populate the `VkDeviceCreateInfo` struct, which requires the `queueCreateInfo` struct. Let's get started:

1.  Create a vector so that we can store `VkDeviceQueueCreateInfo` and any necessary information for the graphics and presentation queues.
2.  Create another vector of the `int` type so that we can store the indices of the graphics and presentation queues.

3.  For each queue family, populate `VkDeviceQueueCreateInfo`. Create a local struct and pass in the struct type, the queue family index, the queue count, and priority (which is `1`), and then push it into the `queueCreateInfos` vector, as shown in the following code:

```cpp
void Device::createLogicalDevice(VkSurfaceKHRsurface, boolisValidationLayersEnabled, AppValidationLayersAndExtensions *appValLayersAndExtentions) { 

   // find queue families like graphics and presentation 
   QueueFamilyIndices indices = findQueueFamilies(physicalDevice, 
          surface); 

   std::vector<VkDeviceQueueCreateInfo> queueCreateInfos; 

   std::set<int> uniqueQueueFamilies = { indices.graphicsFamily, 
                                       indices.presentFamily }; 

   float queuePriority = 1.0f; 

   for (int queueFamily : uniqueQueueFamilies) { 

         VkDeviceQueueCreateInfo queueCreateInfo = {}; 
         queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE
                                 _QUEUE_CREATE_INFO; 
         queueCreateInfo.queueFamilyIndex = queueFamily; 
         queueCreateInfo.queueCount = 1; // we only require 1 queue 
         queueCreateInfo.pQueuePriorities = &queuePriority; 
         queueCreateInfos.push_back(queueCreateInfo); 
   } 
```

4.  To create the device, specify the device features that we will be using. For the device features, we will create a variable of the `VkPhysicalDeviceFeatures` type and set `samplerAnisotropy` to `true`, as follows:

```cpp
 //specify device features  
   VkPhysicalDeviceFeatures deviceFeatures = {};  

   deviceFeatures.samplerAnisotropy = VK_TRUE; 

```

5.  Create the `VkDeviceCreateInfo` struct, which we need in order to create the logical device. Set the type to `VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO` and then set `queueCreateInfos`, the count, and the device features that are to be enabled.

6.  Set the device extension count and names. If the validation layer is enabled, we set the validation layer's count and names. Create the `logicalDevice` by calling `vkCreateDevice` and passing in the physical device, the create device information, and `null` for the allocator. Then, create the logical device, as shown in the following code. If this fails, then we throw a runtime error:

```cpp
   VkDeviceCreateInfo createInfo = {}; 
   createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; 
   createInfo.pQueueCreateInfos = queueCreateInfos.data(); 
   createInfo.queueCreateInfoCount = static_cast<uint32_t>
                                     (queueCreateInfos.size()); 

   createInfo.pEnabledFeatures = &deviceFeatures; 
   createInfo.enabledExtensionCount = static_cast<uint32_t>
     (deviceExtensions.size()); 
   createInfo.ppEnabledExtensionNames = deviceExtensions.data(); 

   if (isValidationLayersEnabled) { 
      createInfo.enabledLayerCount = static_cast<uint32_t>(appValLayersAndExtentions->requiredValidationLayers.size()); 
      createInfo.ppEnabledLayerNames = appValLayersAndExtentions->
                               requiredValidationLayers.data(); 
   } 
   else { 
         createInfo.enabledLayerCount = 0; 
   } 

   //create logical device 

   if (vkCreateDevice(physicalDevice, &createInfo, nullptr, 
      &logicalDevice) != VK_SUCCESS) { 
         throw std::runtime_error("failed to create logical 
            device !"); 
   }
```

7.  Get the device graphics and presentation queue, as shown in the following code. We are now done with the `Device` class:

```cpp
//get handle to the graphics queue of the gpu 
vkGetDeviceQueue(logicalDevice, indices.graphicsFamily, 0, 
&graphicsQueue); 

//get handle to the presentation queue of the gpu 
vkGetDeviceQueue(logicalDevice, indices.presentFamily, 0, &presentQueue); 

}  
```

8.  This wraps up the `Device` class. Include the `Device.h` file in `VulkanContext.h` and add a new device object of the `Device` type to the `VulkanContext` class's private section, as follows:

```cpp
// My Classes
   AppValidationLayersAndExtensions *valLayersAndExt; 
   VulkanInstance* vInstance; 
   Device* device; 
```

9.  In the `VulkanContext.cpp` file in the `VulkanInit` function, add the following code after creating the surface:

```cpp
device = new Device(); 
device->pickPhysicalDevice(vInstance, surface); 
device->createLogicalDevice(surface, isValidationLayersEnabled,
   valLayersAndExt);   
```

10.  This will create a new instance of the `device` class and we choose a device from the available physical devices. You will then be able to create the logical device. Run the application to see which device the application will run on. On my desktop, the following device count and name were found:

![](img/ba1c7ca6-5f59-48e6-81fb-e3ab6cc5abf2.png)

11.  On my laptop, the application found one device with the following device name:

![](img/2c89d883-db8b-4d79-97d1-43caacc381a8.png)

12.  Set breakpoints inside `findQueueFamilies`, `checkDeviceExtensionSupport`, and `querySwapChainSupport` to check for the number of queue family device extensions and for swapchain support for your GPUs.

# Summary

We are about a quarter of the way through the process of seeing something being rendered to the viewport. In this chapter, we set the validation layers and the extension that we will need in order to set up Vulkan rendering. We created a Vulkan application and instance and then created a device class so that we can select the physical device. We also created the logical device so that we can interact with the GPU.

In the next chapter, we will create the swapchain itself so that we can swap between buffers, and we will create the render and the depth texture to draw the scene. We will create a render pass to set how the render textures are to be used and then create the draw command buffers, which will execute our draw commands.