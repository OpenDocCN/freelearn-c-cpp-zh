# Setting Up Your Game

In this chapter, we will start with the basics of how a game is made and what basic graphical components the game requires. Since this book is going to be covering graphics with C++, we will mostly look at what is graphically required from the graphics engine in a game. We will also cover the sound system so that we can make the game more interesting.

To create a basic graphics engine, we will be using the **Simple and Fast Multimedia Library** (**SFML**) since this includes most of the functionality that is needed to get a game up and running. The reason for choosing SFML is that it is very basic and easy to understand, unlike other engines and frameworks.

In this chapter, we will create a window for our game and add animations to it. We will also learn how to create and control our player's movements.

The following topics are covered in this chapter:

*   An overview of SFML
*   Downloading SFML and configuring Visual Studio
*   Creating a window
*   Drawing shapes
*   Adding sprites
*   Keyboard input
*   Handing player movement

# An overview of SFML

Games and video games (unlike other entertainment media) actually involve loading various resources, such as images, videos, sound, font types, and more. SFML provides functions for loading all of these features into games.

SFML is cross-platform compatible, which implies that it permits you to develop and run games on diverse platforms. It also supports various languages other than C++. Additionally, it is open source, so you can take a look at the source code and add a feature to it (if it is not included).

SFML is broken down into five modules, which can be defined as follows:

*   **System**: This module directly interacts with a system such as Windows, which is essentially the **o****perating system** (**OS**) that it will use. Since SFML is cross-platform compatible and each OS is different in terms of how it handles data, this module takes care of interacting with the OS.
*   **Window**: When rendering anything to the screen, the first thing we need is a viewport or a window. Once we have access to this, we can start sending our rendered scene to it. The window module takes care of how a window is created, how the input is handled, and more.
*   **Graphics**: After we have access to a window, we can use the graphics module to begin rendering our scene to the window. In SFML, the graphics module is primarily rendered using OpenGL and deals with 2D scene rendering only. Therefore, it can't be used to make 3D games.
*   **Audio**: The audio module is responsible for playing audio and audio streams, as well as recording audio.
*   **Networking**: SFML also includes a networking library for sending and receiving data, which can be used for developing multiplayer games.

# Downloading SFML and configuring Visual Studio

Now that we are familiar with the basics, let's get started: 

1.  Navigate to the SFML download page ([https://www.sfml-dev.org/download.php](https://www.sfml-dev.org/download.php.)):

![](img/65814d16-760b-4b41-b355-8e269710b1f5.png)

2.  Select **SFML 2.5.1**. Alternatively, you can clone the repository and build the latest version using CMake.
3.  Download either the 32-bit or 64-bit version (depending on your OS) for Visual Studio 2017.

Although we are going to be developing the game for Windows, you can download SFML for Linux or macOS from the same web page.

In the downloaded ZIP file, you will see the following directories:

![](img/cc138488-540c-41c0-b3a3-081fdc183027.png)

These directories can be defined as follows:

*   `bin`: This contains all the **dynamic link libraries** (**DLLs**) that are required for running all the SFML modules. This has a `.dll` file, which contains the debug and release versions. The debug version has a `-d` suffix at the end of the file. Files that don't have this suffix are the release version `.dll` files.
*   `doc`: This contains the documentation for SFML provided in HTML format.
*   `examples`: This contains examples we can use to implement the modules and features of SFML. It tells us how to open a window, include OpenGL, carry out networking, and how to create a basic pong-like game.
*   `include`: This contains all the header files for the modules. The graphics module has classes for creating sprites, loading textures, creating shapes, and more.
*   `lib`: This contains all the library files that we will need in order to run SFML.

There are also two files: `readme.md` and `license.md`. The license file indicates that SFML can be used for commercial purposes. Therefore, it can be altered and redistributed, provided that you don't claim that you created it.

4.  To set up a Visual Studio project, create a new project called `SFMLProject`. In this Visual Studio project root directory, where `SFMLProject.vcxproj` is located, extract the `SFML-2.5.1` folder and place it here.
5.  Then, in the root directory, move all the `.dll` files from the `.bin` folder into the root directory. Your project root directory should look similar to the following screenshot:

![](img/9e2adfcc-5ba7-4ff8-9644-2bc7d108f23e.png)

6.  In the Visual Studio project, create a new `source.cpp` file.
7.  Next, open Project Properties by right-clicking on the project in the Solution Explorer.

8.  Make sure that the Win32 configuration is selected. Under Configuration Properties, select VC++ Directories. Add $(ProjectDir)\SFML-2.5.1\include to the Include Directories. Then, add $(ProjectDIr)\SFML-2.5.1\lib in Library Directories, as shown in the following screenshot:

![](img/80696d8b-ade1-4424-b7af-97c629910851.png)

The `$(ProjectDir)` keyword always makes sure that files are searched with reference to the project directory, which is where the `.vcxproj` file is located. This makes the project portable and able to run on any Windows system.

9.  Next, we have to set what libraries we want to use; select Input from the Linker dropdown menu and type in the following `.lib` files:

![](img/ec65240f-e234-4388-8ff4-8683652e3140.png)

Although we won't be using `sfml-network-d.lib` in this book, it is better to include it so that, if you do want to make a multiplayer game later, then you will already be set up for it.

Now that we've completed the setup, we can finally start typing some code.

# Creating a window

Before we draw anything, the first thing we need is a window so that we can display something on the screen. Let's create a window:

1.  At the top of the `source.cpp` file, include the `Graphics.hpp` file to gain access to the SFML graphics library:

```cpp
#include "SFML-2.5.1\include\SFML\Graphics.hpp"  
```

2.  Next, add the main function, which will be the application's main entry point:

```cpp
int main(){ 

return 0; 
}  
```

3.  To create the window, we have to specify the size of the window that we want to create. SFML has a `Vector2f` data type, which takes an `x` and a `y` value and uses them to define the size of the window that we will be using.

Between `include` and `main`, add the following line of code. Create a variable called `viewSize` and set the `x` and `y` values to `1024` and `768`, respectively:

```cpp
sf::Vector2f viewSize(1024, 768); 
```

The assets for the game are created for the resolution, so I am using this view size; we also need to specify a `viewMode` class.

`viewMode` is an SFML class that sets the width and height of the window. It also gets the bits that are required to represent a color value in a pixel. `viewMode` also obtains the different resolutions that your monitor supports so that you can let the user set the resolution of the game to glorious 4K resolution if they desire.

4.  To set the view mode, add the following code after setting the `viewSize` variable:

```cpp
sf::videoMode vm(viewSize.x, viewSize.y);  
```

5.  Now, we can finally create a window. The window is created using the `RenderWindow` class. The `RenderWindow` constructor takes three parameters: a `viewMode` parameter, a window name parameter, and a `Style` parameter.

We have already created a `viewMode` parameter and we can pass in a window name here using a string. The third parameter, `Style`. `Style`, is an `enum` value; we can add a number of values, called a bitmask, to create the window style that we want:

*   `sf::style::Titlebar`: This adds a title bar to the top of the window.
*   `sf::style::Fullscreen`: This creates a full-screen mode window.
*   `sf::style::Default`: This is the default style that combines the ability to resize a window, close it, and add a title bar.

6.  Let's create a default-style window. First, create the window using the following command and add it after the creation of the `viewMode` parameter:

```cpp
sf::RenderWindow window(vm, "Hello SFMLGame !!!", sf::Style::Default); 
```

7.  In the `main()` class, we will create a `while` loop, which handles the main game loop for our game. This will check whether or not the window is open so that we can add some keyboard events by updating and rendering the objects in the scene. The `while` loop will run as long as the window is open. In the `main` function, add the following code:

```cpp
int main() { 

   //Initialize Game Objects 

         while (window.isOpen()) { 

               // Handle Keyboard Events 
               // Update Game Objects in the scene 
               // Render Game Objects  

         } 

   return 0; 
}
```

Now, run the application. Here, you have your not-so-interesting game with a window that has a white background. Hey, at least you have a window now! To display something here, we have to clear the window and display whatever we draw in every frame.  This is done using the `clear` and `display` functions.

8.  We have to call `window.clear()` before we can render the scene and then call `window.display()` afterward to display the scene objects.

In the `while` loop, add the `clear` and `display` functions. Game objects will be drawn between the `clear` function and the `display` function:

```cpp
int main() { 

   //init game objects 
         while (window.isOpen()) { 
               // Handle Keyboard events 
               // Update Game Objects in the scene 
    window.clear(sf::Color::Red);                
 // Render Game Objects  
window.display();
         } 
   return 0; 
} 
```

The `clear` function takes in a clear color. Here, we are passing in the color red as a value into the function. This function fills in the whole window with this solid color value: 

![](img/4202f8f9-7655-403d-beab-f571306cec60.png)

# Drawing shapes

SFML provides us with the functionality to draw basic shapes such as a rectangle, circle, and triangle. The shape can be set to a certain size and has functions, such as `fillColor`, `Position`, and `Origin`, so that we can set the color, the position of the shape in the viewport, and the origin around which the shape can rotate respectively. Let's take a look at an example of a rectangular shape:

1.  Before the `while` loop, add the following code to set up the rectangle:

```cpp

   sf::RectangleShape rect(sf::Vector2f(500.0f, 300.0f)); 
   rect.setFillColor(sf::Color::Yellow); 
   rect.setPosition(viewSize.x / 2, viewSize.y / 2); 
   rect.setOrigin(sf::Vector2f(rect.getSize().x / 2, 
   rect.getSize().y / 2)); 

```

Here, we created a `Rectangle` parameter of the `RectangleShape` type and named it `rect`. The constructor of `RectangleShape` takes in the size of the rectangle. Its size is `500` by `300`. Then, we set the color of the rectangle to yellow. After this, we set the position of the rectangle to the center of the viewport and set the origin to the center of the rectangle.

2.  To draw the rectangle, we have to call the `window.draw()` function and pass the rectangle into it. Make sure that you call this function between the `clear` and `display` functions in the `while` loop. Now, add the following code:

```cpp

   #include "SFML-2.5.1\include\SFML\Graphics.hpp" 

   sf::Vector2f viewSize(1024, 768); 
   sf::VideoMode vm(viewSize.x, viewSize.y); 
   sf::RenderWindow window(vm, "Hello Game SFML !!!", 
   sf::Style::Default); 

   int main() { 

   //init game objects 

   sf::RectangleShape rect(sf::Vector2f(500.0f, 300.0f));
 rect.setFillColor(sf::Color::Yellow);
 rect.setPosition(viewSize.x / 2, viewSize.y / 2);
 rect.setOrigin(sf::Vector2f(rect.getSize().x / 2, 
   rect.getSize().y / 2));
         while (window.isOpen()) { 
               // Handle Keyboard events 
               // Update Game Objects in the scene 

               window.clear(sf::Color::Red); 

               // Render Game Objects  
               window.draw(rect);

               window.display(); 
         } 
   return 0; 
} 
```

3.  Now, run the project; you will see a yellow rectangle in a red viewport, as follows:

![](img/9efb88d6-b2f6-4ae1-a98f-9ab5d9f53a25.png)

4.  If we set the position to `(0, 0)`, you will see where the origin is for the 2D rectangle in SFML—it is in the top left corner of the viewport:

![](img/2ffda409-1467-4cde-87fe-8d5ce8fa778e.png)

5.  Move it back to the center of the viewport by undoing the previous action. Then, set the rectangle to the center of the viewport again, as follows:

```cpp
rect.setPosition(viewSize.x / 2, viewSize.y / 2);
```

6.  Now, we can add a few more shapes, such as a circle and a triangle. We can create a circle using the `CircleShape` class, whereas we can create a triangle using the `ConvexShape` class. Before the main loop, we will create a circle by using `CircleShape` and `Triangle` with `ConvexShape`, as follows:

```cpp
   sf::CircleShape circle(100); 
   circle.setFillColor(sf::Color::Green); 
   circle.setPosition(viewSize.x / 2, viewSize.y / 2); 
   circle.setOrigin(sf::Vector2f(circle.getRadius(), 
   circle.getRadius())); 
   sf::ConvexShape triangle; 
   triangle.setPointCount(3); 
   triangle.setPoint(0, sf::Vector2f(-100, 0)); 
   triangle.setPoint(1, sf::Vector2f(0, -100)); 
   triangle.setPoint(2, sf::Vector2f(100, 0)); 
   triangle.setFillColor(sf::Color(128, 0, 128, 255)); 
   triangle.setPosition(viewSize.x / 2, viewSize.y / 2); 
```

The `CircleShape` class takes only one parameter (which is the radius of the circle), in comparison to the rectangle, which takes two parameters. We set the color of the circle to green using the `setFIllColor` function, and then set its position and origin.

To create the triangle, we use the `ConvexShape` class. To create a shape, we specify the `setPointCount`, which takes one parameter. We will use it to specify how many points will make up the shape. Next, using the `setPoint` function, we set the location of the points. This takes two parameters: the first is the index of the point and the second is the location of the point.

To create the triangle, we use three points: the first, with an index of `0` and a location of `(-100, 0)`; the second, with an index of `1` and a location of `(0, -100)`; and the third, with an index of `2` and a location of `(100, 0)`.

Now, we need to set the color of the triangle. We do this by setting the values of the red, green, blue, and alpha values. Colors in SFML are 8-bit integer values. This means that each color range is between 0 and 255, where 0 is black and 255 is the maximum color range. So, when we set the color of the triangle to `triangle.setFillColor(sf::Color(128, 0, 128, 255));`, red is half of its maximum range, there is no green, blue is also half of its maximum range, and alpha is `255`, making the triangle fully opaque. Then, we set the position of the triangle so that it's at the center of the screen.

7.  Next, we draw the circle and triangle. Call the `draw` function for the circle and triangle after drawing the rectangle, as follows:

```cpp
  while (window.isOpen()) { 
               // Handle Keyboard events 
               // Update Game Objects in the scene 

               window.clear(sf::Color::Red); 

               // Render Game Objects  
               window.draw(rect); 
               window.draw(circle);
 window.draw(triangle);

               window.display(); 
         } 
```

8.  The output of the preceding code is as follows:

![](img/bc59e484-d415-4dac-9c27-8baf8f7f5a2b.png)

Note that, when creating the triangle, the second point was created with a negative *y* value of `100`:

```cpp
triangle.setPoint(0, sf::Vector2f(-100, 0)); 
triangle.setPoint(1, sf::Vector2f(0, -100)); 
triangle.setPoint(2, sf::Vector2f(100, 0)); 
```

However, the triangle is pointing upward. This means that the *+y* axis is pointing downward. You will find that this is mostly the case in 2D frameworks. Furthermore, the origin for the scene is in the top-left corner, so the coordinate system is as follows:

![](img/cfe38702-fcdf-4381-b183-15686c0cd830.png)

It is also important to note that the drawing order matters. Drawing happens from back to front. So, the first shape that's drawn will be behind the shapes that are drawn later in the same location. Objects that are drawn later simply draw over the earlier objects, in the same way that an artist would draw in real life when painting on a canvas. So, make sure that you draw the bigger objects first and then draw the smaller ones later. If you draw the smaller objects before the bigger ones, then the smaller objects will be behind the bigger objects and you won't be able to see them. Make sure this doesn't happen as you won't get any errors and everything in the code will be correct, so you won't know if something's gone wrong.

# Adding sprites

A sprite is a rectangle with a picture applied to it. You may be wondering, *why not just use a picture?* Of course, we do load a picture up, then we won't be able to move or rotate it. Therefore, we apply a picture or texture to a rectangle that is able to move and rotate, making it look as if the picture is doing so. Let's learn how to do this:

1.  Since we will be loading images into our game project, which is in the root directory of the project, let's create a folder called `Assets`.
2.  In this folder, create another folder called `graphics`, and then copy and paste the `sky.png` file into the `graphics` folder:

![](img/73547c73-e482-4051-8ea6-4255b56d8cba.png)

To create sprites, we use the `Sprite` class from SFML. The `Sprite` class takes in a texture. Then, the picture is loaded using the `Texture` class. While drawing, you need to call `window.draw.(sprite)` to draw the sprite. Let's take a look at how to do this.

3.  Declare a `Texture` class called `skyTexture` and a `Sprite` class called `skySprite` globally. This should be done after the creation of the `RenderWindow` class:

```cpp
sf::Texture skyTexture; 
sf::Sprite skySprite;
```

4.  Create a new function called `init` in the `source.cpp` file that appears right before the `main` function. Since we don't want the `main` function to be cluttered, we will add the code to initialize `skyTexture` and `skySprite` to it. In the `init` function, add the following code:

```cpp
void init() { 

// Load sky Texture 
   skyTexture.loadFromFile("Assets/graphics/sky.png"); 

// Set and  Attacha Texture to Sprite 
   skySprite.setTexture(skyTexture); 

} 
```

First, we load the `skyTexture` function by calling the `loadFromFile` function. We pass in the path and filename of the file that we want to load. Here, we want to load the `sky.png` file from the `Assets` folder.

Next, we use the `setTexture` function of the sprite and pass the `skyTexture` function into it.

5.  To do this, create a new function called `draw()` above the `main` and `init` functions. We call draw (`skySprite`) in it in order to draw the sprite, as follows:

```cpp
void draw() { 

   window.draw(skySprite); 

} 
```

6.  Now, we have to call `init()` at the beginning of the `main` function and `draw()` in the `while` loop that we added to the `main` function. You can remove all the code that was used for creating and drawing the shapes from the `main` function. Your `main` function should look as follows:

```cpp
 #include "SFML-2.5.1\include\SFML\Graphics.hpp" 

sf::Vector2f viewSize(1024, 768); 
sf::VideoMode vm(viewSize.x, viewSize.y); 
sf::RenderWindow window(vm, "Hello Game SFML !!!", sf::Style::Default); 

sf::Texture skyTexture; 
sf::Sprite skySprite; 

void init() { 

   skyTexture.loadFromFile("Assets/graphics/sky.png"); 
   skySprite.setTexture(skyTexture); 

} 

void draw() { 

   window.draw(skySprite); 

} 

int main() { 

   init(); 

   while (window.isOpen()) { 

         window.clear(sf::Color::Red); 

         draw(); 

         window.display(); 

   } 

   return 0; 
} 
```

The output is as follows:

![](img/afd3616f-cc49-4c28-a5e2-d82b246818d4.png)

Praise the sun! Lo and behold, we have the sky texture loaded and have drawn it as a sprite in the window.

7.  I have included a background texture picture as well, called `bg.png`, which is available in the `Assets` folder of this chapter's project. Try and load the texture and draw the texture in the same way.
8.  I named the variables of the background texture and sprite `bgTexture` and `bgSprite`, respectively, and drew the `bgSprite` variable into the scene. Don't forget to add the `bg.png` file to the `Assets/graphics` directory.

Your scene should now look as follows:

![](img/db72e296-c03e-40bd-a81b-14fe9ddf80fe.png)

9.  Next, add another sprite called `heroSprite` and load in the picture with `heroTexture`. Set the origin of the sprite to its center and place it in the middle of the scene. The `hero.png` file image is provided here, so make sure you place it in the `Assets/graphics` folder. Now, declare `heroSprite` and `heroTexture`, as follows:

```cpp
sf::Texture heroTexture; 
sf::Sprite heroSprite; 

In the init function initialize the following values: 
   heroTexture.loadFromFile("Assets/graphics/hero.png"); 
   heroSprite.setTexture(heroTexture); 
   heroSprite.setPosition(sf::Vector2f(viewSize.x/2, 
      viewSize.y/2)); 
   heroSprite.setOrigin(heroTexture.getSize().x / 2, 
      heroTexture.getSize().y / 2);  
```

10.  To set the origin of the sprite, we take the textures and the height and divide them by `2`.

Using the `draw` function, draw the `heroSprite` sprite, as follows:

```cpp
void draw() { 
   window.draw(skySprite); 
   window.draw(bgSprite); 
   window.draw(heroSprite);
}
```

11.  Our hero will now appear in the scene, as follows:

![](img/2dfe1e8c-a587-4105-b894-c837d2da6aae.png)

# Keyboard input

It is great that we are able to add shapes, sprites, and textures; however, computer games, by nature, are interactive. We will need to allow players to use keyboard inputs so that they can access the game's content. But how do we know which button the player is pressing? Well, that is handled through the polling of events. Polling just checks the status of the keys regularly; events are used to check whether an event was triggered, such as the closing of the viewport.

SFML provides the `sf::Event` class so that we can poll events. We can use the `pollEvent` function of the window to check for events that may be occurring, such as a player pressing a button.

Create a new function called `updateInput()`. Here, we will create a new object of the `sf::Event` class called `event`. We will create a `while` loop called `window.pollEvent` and then pass in the `event` variable to check for events.

So far, we have been using *Shift* + *F5* or the stop button in Visual Studio to stop the application. One of the basic things we can do is check whether the *Esc* key has been pressed. If it has been pressed, we want to close the window. To do this, add the following code:

```cpp
void updateInput() { 

   sf::Event event; 

   while (window.pollEvent(event)) { 

if (event.key.code == sf::Keyboard::Escape || 
         event.type ==sf::Event::Closed) 
                     window.close(); 
   } 

} 
```

In the `while` loop, we need to check whether the event key code is that of the *Esc* key code, or whether the event is `Event::closed`. Then, we call the `window.close()` function to close the window. When we close the window, it shuts down the application.

Call the `updateInput()` function in the main `while` loop before the `window.clear()` function. Now, when you press *Esc* while the application is running, it will close. SFML doesn't limit inputs just to keyboards; it also provides functionality for mouse, joystick, and touch input.

# Handing player movement

Now that we have access to the player's keyboard, we can learn how to move game objects. Let's move the player character to the right when the right arrow key is pressed on the keyboard. We will stop moving the hero when the right arrow key is released:

1.  Create a global `Vector2f` called `playerPosition`, right after `heroSprite`.
2.  Create a Boolean data type called `playerMoving` and set it to `false`.
3.  In the `updateInput` function, we will check whether the right key has been pressed or released. If the button is pressed, we set `playerMoving` to `true`. If the button is released, then we set `playerMoving` to `false`.

The `updateInput` function should be as follows:

```cpp
void updateInput() { 

   sf::Event event; 

   while (window.pollEvent(event)) { 

         if (event.type == sf::Event::KeyPressed) { 

               if (event.key.code == sf::Keyboard::Right) { 

                     playerMoving = true; 
               } 
         }           
         if (event.type == sf::Event::KeyReleased) { 

               if (event.key.code == sf::Keyboard::Right) { 
                     playerMoving = false; 
               }               
         } 

         if (event.key.code == sf::Keyboard::Escape || event.type 
         == sf::Event::Closed) 
               window.close();  
   } 
}
```

4.  To update the objects in the scene, we will create a function called `update`, which will take a float called `dt`. This stands for delta time and refers to the time that has passed between the previous update and the current update call. In the `update` function, we will check whether the player is moving. If the player is moving, then we will move the position of the player in the *+x* direction and multiply this by `dt`.

The reason we multiply by delta time is because if we don't, then the update will not be time-dependent, but processor-dependent instead. If you don't multiply the position by `dt`, then the update will happen faster on a faster PC and will be slower on a slower PC. So, make sure that any movement is always multiplied by `dt`.

The `update` function should look as follows. Make sure that this function appears before the `main` function:

```cpp
void update(float dt) { 

   if (playerMoving) { 
         heroSprite.move(50.0f * dt, 0); 
   } 
} 
```

5.  At the beginning of the `main` function, create an object of the `sf::Clock` type called `Clock`. The `Clock` class takes care of getting the system clock and allows us to get the delta time in seconds, milliseconds, or microseconds.
6.  In the `while` loop, after calling `updateInput()`, create a variable called `dt` of the `sf::Time` type and set the `dt` variable by calling `clock.restart().`
7.  Now, call the `update` function and pass in `dt.asSeconds()`, which will give the delta time as 60 frames per second, which is approximately .0167 seconds.

The `main` function should appear as follows:

```cpp

int main() { 

   sf::Clock clock; 
   init(); 
   while (window.isOpen()) { 

         // Update input 
         updateInput(); 

         // Update Game 
         sf::Time dt = clock.restart(); 
         update(dt.asSeconds()); 

   window.clear(sf::Color::Red); 

         //Draw Game  
         draw(); 

         window.display();   
   }  
   return 0; 
} 
```

8.  Now, when you run the project and press the right arrow key on the keyboard, the player will start moving right, and will stop when you release the right arrow key:

![](img/44a97710-1da7-4ba3-8fb2-3ee7edef437b.png)

# Summary

In this chapter, we looked at how to set up SFML so that we can start creating a game. We covered the five basic modules that make up SFML, and also looked at creating shapes using SFML and adding the background and player sprite to the scene. We also added keyboard input and used this to make the player character move within the scene.

In the next chapter, we will create the basic skeleton of the game. We will also move the player character to a separate class and add some basic physics to the character to allow them to jump in the game.