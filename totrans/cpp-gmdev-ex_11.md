# Enhancing Your Game with Collision, Loops, and Lighting

In this chapter, we will learn how to add collision to detect contact between the ball and the enemy; this will determine the lose condition. We will also check the contact between the ball and the ground to find out whether the player can jump or not. Then, we will finalize the gameplay loop.

Once the gameplay loop is complete, we will be able to add text rendering to show the player their score. To display the necessary text, we will use the FreeType library. This will load in the characters from the font file.

We will also add some basic lighting to the objects in the scene. Lighting will be calculated using the Phong lighting model, and we will cover how this is implemented in practice. To finish the gameplay loop, we will have to add an enemy.

In this chapter, we will cover the following topics:

*   Adding a `RigidBody` name
*   Adding an enemy
*   Moving the enemy
*   Checking collision
*   Adding keyboard controls
*   Gameloop and scoring
*   Text rendering
*   Adding lighting

# Adding a RigidBody name

To identify the different rigid bodies we are going to be adding to the scene, we will add a property to the `MeshRenderer` class that will specify each object being rendered. Let's look at how to do this:

1.  In the `MeshRenderer.h` class, which can be found within the `MeshRenderer` class, change the constructor of the class to take in a string as the name for the object, as follows:

```cpp
MeshRenderer(MeshType modelType, std::string _name, Camera *  
   _camera, btRigidBody* _rigidBody) 
```

2.  Add a new public property called `name` of the `std::string` type and initialize it, as follows:

```cpp
         std::string name = ""; 
```

3.  Next, in the `MeshRenderer.cpp` file, modify the constructor implementation, as follows:

```cpp
MeshRenderer::MeshRenderer(MeshType modelType, std::string _name,  
   Camera* _camera, btRigidBody* _rigidBody){ 

   name = _name; 
... 
... 

} 
```

We have successfully added the `name` property to the `MeshRenderer` class.

# Adding an enemy

Before we add an enemy to the scene, let's clean up our code a little bit and create a new function called `addRigidBodies` in `main.cpp` so that all the rigid bodies will be created in a single function. To do so, follow these steps:

1.  In the source of the `main.cpp` file, create a new function called `addRigidBodies` above the `main()` function.

2.  Add the following code to the `addRigidBodies` function. This will add the sphere and ground. We are doing this instead of putting all the game code in the `main()` function:

```cpp
   // Sphere Rigid Body 

   btCollisionShape* sphereShape = new btSphereShape(1); 
   btDefaultMotionState* sphereMotionState = new 
     btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), 
     btVector3(0, 0.5, 0))); 

   btScalar mass = 13.0f; 
   btVector3 sphereInertia(0, 0, 0); 
   sphereShape->calculateLocalInertia(mass, sphereInertia); 

   btRigidBody::btRigidBodyConstructionInfo sphereRigidBodyCI(mass, 
      sphereMotionState, sphereShape, sphereInertia); 

   btRigidBody* sphereRigidBody = new btRigidBody(
                                  sphereRigidBodyCI); 

   sphereRigidBody->setFriction(1.0f); 
   sphereRigidBody->setRestitution(0.0f); 

   sphereRigidBody->setActivationState(DISABLE_DEACTIVATION); 

   dynamicsWorld->addRigidBody(sphereRigidBody); 

   // Sphere Mesh 

   sphere = new MeshRenderer(MeshType::kSphere, "hero", camera, 
            sphereRigidBody); 
   sphere->setProgram(texturedShaderProgram); 
   sphere->setTexture(sphereTexture); 
   sphere->setScale(glm::vec3(1.0f)); 

   sphereRigidBody->setUserPointer(sphere); 

   // Ground Rigid body 

   btCollisionShape* groundShape = new btBoxShape(btVector3(4.0f, 
                                   0.5f, 4.0f)); 
   btDefaultMotionState* groundMotionState = new 
       btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), 
       btVector3(0, -1.0f, 0))); 

   btRigidBody::btRigidBodyConstructionInfo groundRigidBodyCI(0.0f, 
      groundMotionState, groundShape, btVector3(0, 0, 0)); 

   btRigidBody* groundRigidBody = new btRigidBody(
                                  groundRigidBodyCI); 

   groundRigidBody->setFriction(1.0); 
   groundRigidBody->setRestitution(0.0); 

   groundRigidBody->setCollisionFlags(
       btCollisionObject::CF_STATIC_OBJECT); 

   dynamicsWorld->addRigidBody(groundRigidBody); 

   // Ground Mesh 
   ground = new MeshRenderer(MeshType::kCube, "ground", camera, 
            groundRigidBody); 
   ground->setProgram(texturedShaderProgram); 
   ground->setTexture(groundTexture); 
   ground->setScale(glm::vec3(4.0f, 0.5f, 4.0f)); 

   groundRigidBody->setUserPointer(ground); 

```

Note that some of the values have been changed to suit our game. We have also disabled deactivation on the sphere because, if we don't, then the sphere will be unresponsive when we want it to jump for us.

To access the name of the rendered mesh, we can set this instance as a property of the rigid body by using the `setUserPointer` property of the `RigidBody` class. `setUserPointer` takes a void pointer, so any kind of data can be passed into it. For the sake of convenience, we are just passing the instance of the `MeshRenderer` class itself. In this function, we will also add the enemy's rigid body to the scene, as follows:

```cpp
// Enemy Rigid body 

btCollisionShape* shape = new btBoxShape(btVector3(1.0f, 1.0f, 1.0f)); 
btDefaultMotionState* motionState = new btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), 
btVector3(18.0, 1.0f, 0))); 
btRigidBody::btRigidBodyConstructionInfo rbCI(0.0f, motionState, shape, btVector3(0.0f, 0.0f, 0.0f)); 

   btRigidBody* rb = new btRigidBody(rbCI); 

   rb->setFriction(1.0); 
   rb->setRestitution(0.0); 

//rb->setCollisionFlags(btCollisionObject::CF_KINEMATIC_OBJECT); 

rb->setCollisionFlags(btCollisionObject::CF_NO_CONTACT_RESPONSE); 

   dynamicsWorld->addRigidBody(rb); 

   // Enemy Mesh 
   enemy = new MeshRenderer(MeshType::kCube, "enemy", camera, rb); 
   enemy->setProgram(texturedShaderProgram); 
   enemy->setTexture(groundTexture); 
   enemy->setScale(glm::vec3(1.0f, 1.0f, 1.0f)); 

   rb->setUserPointer(enemy); 
```

3.  Add the enemy in the same way that we added the sphere and the ground. Since the shape of the enemy object is a cube, we use`btBoxShape` to set the shape of the box for the rigid body. We set the location to 18 units' distance in the *X*-axis and one unit's distance in the *Y*-axis. Then, we set the friction and restitution values.

For the type of the rigid body, we set its collision flag to `NO_CONTACT_RESPONSE` instead of `KINEMATIC_OBJECT`. We could have set the type to `KINEMATIC_OBJECT`, but then the enemy object would exert force on other objects, such as the sphere, when it comes in contact with it. To avoid this, we use `NO_CONTACT_RESPONSE`, which will just check if there was an overlap between the enemy rigid body and another body, instead of applying force to it.

You can uncomment the `KINEMATIC_OBJECT` line of code and comment on the `NO_CONTACT_RESPONSE` line of code to see how using either changes the way the object behaves in the physics simulation.

4.  Once we have created the rigid body, we add the rigid body to the world, set the mesh renderer for the enemy object, and name it **enemy**.

# Moving the enemy

To update the enemy's movement, we will add a tick function that will be called by the rigid body world. In this tick function, we will update the position of the enemy so that the enemy cube moves from the right of the screen to the left. We will also check whether the enemy has gone beyond the left-hand side of the screen.

If it has, then we will reset its position to the right of the screen. To do so, follow these steps:

1.  In this update function, we will also update our gameplay logic and scoring, as well as how we check for contact between the sphere and the enemy and the sphere and the ground. Add the tick function callback prototype to the top of the `Main.cpp` file, as follows:

```cpp
   void myTickCallback(btDynamicsWorld *dynamicsWorld, 
      btScalar timeStep); 
```

2.  In the `TickCallback` function, update the position of the enemy, as follows:

```cpp
void myTickCallback(btDynamicsWorld *dynamicsWorld, btScalar timeStep) { 

         // Get enemy transform 
         btTransform t(enemy->rigidBody->getWorldTransform()); 

         // Set enemy position 
         t.setOrigin(t.getOrigin() + btVector3(-15, 0, 0) * 
         timeStep); 

         // Check if offScreen 
         if(t.getOrigin().x() <= -18.0f) { 
               t.setOrigin(btVector3(18, 1, 0)); 
         } 
         enemy->rigidBody->setWorldTransform(t); 
         enemy->rigidBody->getMotionState()->setWorldTransform(t); 

} 
```

In the `myTickCallback` function, we get the current transform and store it in a variable, `t`. Then, we set the origin, which is the position of the transform, by getting the current position, moving it 15 units to the left, and multiplying it by the current timestep (which is the difference between the previous and current time).

Once we get the updated location, we check that the current location is less than 18 units. If it is, then the current location is beyond the screen bounds on the left of the screen. Consequently, we set the current location back to the right of the viewport and make the object wrap around the screen.

Then, we update the location of the object itself to this new location by updating the `worldTransform` of the rigid body and the motion state of the object.

3.  Set the tick function as the default `TickCallback` of the dynamic world in the `init` function, as follows:

```cpp
dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, 
                solver, collisionConfiguration); 
dynamicsWorld->setGravity(btVector3(0, -9.8f, 0));  
dynamicsWorld->setInternalTickCallback(myTickCallback); 
```

4.  Build and run the project to see the cube enemy spawn at the right of the screen, followed by it passing through the sphere and moving toward the left of the screen. When the enemy goes offscreen, it will be looped around to the right of the screen, as shown in the following screenshot:

![](img/ec234c82-a108-4331-bfa9-e8c8fe7eab04.png)

5.  If we set the `collisionFlag` of the enemy to `KINEMATIC_OBJECT`, you will see that the enemy doesn't go through the sphere but pushes it off the ground, as follows:

![](img/ab66fea4-7ddd-4c3d-b00c-caf95c27b182.png)

6.  This is not what we want as we don't want the enemy to physically interact with any objects. Change the collision flag of the enemy back to `NO_CONTACT_RESPONSE` to amend this.

# Checking collision

In the tick function, we need to check for collision between the sphere and the enemy, as well as the sphere and the ground. Follow these steps to do so:

1.  To check the number of contacts between objects, we will use the `getNumManifolds` property of the dynamic world object. The manifold will contain information regarding all the contacts in the scene per update cycle.
2.  We need to check whether the number of contacts is greater than zero. If it is, then we check which pairs of objects were in contact with each other. After updating the enemy object, add the following code to check for contact between the hero and the enemy:

```cpp
int numManifolds = dynamicsWorld->getDispatcher()->
  getNumManifolds(); 

   for (int i = 0; i < numManifolds; i++) { 

       btPersistentManifold *contactManifold = dynamicsWorld->
       getDispatcher()->getManifoldByIndexInternal(i); 

       int numContacts = contactManifold->getNumContacts(); 

       if (numContacts > 0) { 

           const btCollisionObject *objA = contactManifold->
           getBody0(); 
           const btCollisionObject *objB = contactManifold->
           getBody1(); 

           MeshRenderer* gModA = (MeshRenderer*)objA->
           getUserPointer(); 
           MeshRenderer* gModB = (MeshRenderer*)objB->
           getUserPointer(); 

                if ((gModA->name == "hero" && gModB->name == 
                  "enemy") || (gModA->name == "enemy" && gModB->
                  name == "hero")) { 
                        printf("collision: %s with %s \n",
                        gModA->name, gModB->name); 

                         if (gModB->name == "enemy") { 
                             btTransform b(gModB->rigidBody-
                             >getWorldTransform()); 
                             b.setOrigin(btVector3(18, 1, 0)); 
                             gModB->rigidBody-
                             >setWorldTransform(b); 
                             gModB->rigidBody-> 
                             getMotionState()-
                             >setWorldTransform(b); 
                           }else { 

                                 btTransform a(gModA->rigidBody->
                                 getWorldTransform()); 
                                 a.setOrigin(btVector3(18, 1, 0)); 
                                 gModA->rigidBody->
                                 setWorldTransform(a); 
                                 gModA->rigidBody->
                                 getMotionState()->
                                 setWorldTransform(a); 
                           } 

                     } 

                     if ((gModA->name == "hero" && gModB->name == 
                         "ground") || (gModA->name == "ground" &&               
                          gModB->name  == "hero")) { 
                           printf("collision: %s with %s \n",
                           gModA->name, gModB->name); 

                     } 
         } 
   } 
```

3.  First, we get the number of contact manifolds or contact pairs. Then, for each contact manifold, we check whether the number of contacts is greater than zero. If it is greater than zero, then it means there has been a contact in the current update.

4.  Then, we get both collision objects and assign them to `ObjA` and `ObjB`. After this, we get the user pointer for both objects and typecast it to `MeshRenderer` to access the name of the objects we assigned. When checking for contact between two objects, object A can be in contact with object B or the other way around. If there has been contact between the sphere and the enemy, we set the position of the enemy back to the right of the viewport. We also check for contact between the sphere and the ground. If there is contact, we just print out that there has been contact.

# Adding keyboard controls

Let's add some keyboard controls so that we can interact with the sphere. We will set it so that, when we press the up key on the keyboard, the sphere jumps. We will add the jump feature by applying an impulse to the sphere. To do so, follow these steps:

1.  First, we'll use `GLFW`, which has a keyboard callback function so that we can add interaction with the keyboard for the game. Before we begin with the `main()` function, we will set this keyboard callback  function:

```cpp
void updateKeyboard(GLFWwindow* window, int key, int scancode, int action, int mods){ 

   if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { 
         glfwSetWindowShouldClose(window, true);    
   } 

   if (key == GLFW_KEY_UP && action == GLFW_PRESS) { 
               if (grounded == true) { 
                     grounded = false; 

sphere->rigidBody->applyImpulse(btVector3(0.0f, 
   100.0f, 0.0f), btVector3(0.0f, 0.0f, 0.0f)); 
                     printf("pressed up key \n"); 
               } 
         } 
} 
```

The two main parameters that we are concerned with are the key and action. With key, we get which key is pressed, and with action, we can retrieve what action was performed on that key. In the function, we check whether the *Esc* key was pressed using the `glfwGetKey` function. If so, then we close the window using the `glfwSetWindowShouldClose` function by passing true as the second parameter.

To make the sphere jump, we check whether the up key was pressed. If it was, we create a new Boolean member variable called `grounded`, which describes a state if the sphere is touching the ground. If this is true, we set the Boolean value to `false` and apply an impulse of `100` units on the sphere's rigid body origin in the Y direction by calling the `applyImpulse` function of `rigidbody`.

2.  In the tick function, before we get the number of manifolds, we set the `grounded` Boolean to false, as follows:

```cpp
 grounded = false; 

   int numManifolds = dynamicsWorld->getDispatcher()->
                      getNumManifolds(); 
```

3.  We set the `grounded` Boolean value to true when there is contact between the sphere and the ground, as follows:

```cpp
   if ((gModA->name == "hero" && gModB->name == "ground") || 
         (gModA->name == "ground" && gModB->name == "hero")) { 

//printf("collision: %s with %s \n", gModA->name, gModB->name); 

         grounded = true; 

   }   
```

4.  In the main function, set `updateKeyboard` as the callback using `glfwSetKeyCallback`, as follows:

```cpp
int main(int argc, char **argv) { 
...       
   glfwMakeContextCurrent(window); 
   glfwSetKeyCallback(window, updateKeyboard); 
   ... 
   }
```

5.  Now, build and run the application. Press the up key to see the sphere jump, but only when it is grounded, as follows:

![](img/e9392898-a6a3-4ddf-ae97-46d496b357c3.png)

# Game loop and scoring

Let's wrap this up by adding scoring and finishing the game loop:

1.  Along with the `grounded` Boolean, add another Boolean and check for `gameover`. After doing this, add an `int` called `score` and initialize it to `0` at the top of the `main.cpp` file, as follows:

```cpp
GLuint sphereTexture, groundTexture; 

bool grounded = false; 
bool gameover = true; 
int score = 0; 

```

2.  Next, in the tick function, the enemy should only move when the game is not over. So we wrap the update for the position of the enemy inside an **if** statement to check whether or not the game is over. If the game is not over, then we update the position of the enemy, as follows:

```cpp
void myTickCallback(btDynamicsWorld *dynamicsWorld, btScalar timeStep) { 

   if (!gameover) { 

         // Get enemy transform 
         btTransform t(enemy->rigidBody->getWorldTransform()); 

         // Set enemy position 

         t.setOrigin(t.getOrigin() + btVector3(-15, 0, 0) * 
         timeStep); 

         // Check if offScreen 

         if (t.getOrigin().x() <= -18.0f) { 

               t.setOrigin(btVector3(18, 1, 0)); 
               score++; 
               label->setText("Score: " + std::to_string(score)); 

         } 

         enemy->rigidBody->setWorldTransform(t); 
         enemy->rigidBody->getMotionState()->setWorldTransform(t); 
   } 
... 
} 
```

3.  We also increment the score if the enemy goes beyond the left of the screen. Still in the tick function, if there is contact between the sphere and the enemy, we set the score to `0` and set `gameover` to `true`, as follows:

```cpp

         if ((gModA->name == "hero" && gModB->name == "enemy") || 
                    (gModA->name == "enemy" && gModB->name ==
                     "hero")) { 

                     if (gModB->name == "enemy") { 
                         btTransform b(gModB->rigidBody->
                         getWorldTransform()); 
                         b.setOrigin(btVector3(18, 1, 0)); 
                         gModB->rigidBody->
                         setWorldTransform(b); 
                         gModB->rigidBody->getMotionState()->
                         setWorldTransform(b); 
                           }else { 

                           btTransform a(gModA->rigidBody->
                           getWorldTransform()); 
                           a.setOrigin(btVector3(18, 1, 0)); 
                           gModA->rigidBody->
                           setWorldTransform(a); 
                           gModA->rigidBody->getMotionState()->
                           setWorldTransform(a); 
                           } 

                           gameover = true; 
                           score = 0; 

                     }   
```

4.  In the update keyboard function, when the up keyboard key is pressed, we check whether the game is over. If it is, we set the `gameover` Boolean to false, which will start the game. Now, when the player presses the up key again, the character will jump. This way, the same key can be used for starting the game and also making the character jump.
5.  Make the required changes to the `updateKeyboard` function, as follows:

```cpp
void updateKeyboard(GLFWwindow* window, int key, int scancode, int action, int mods){ 

   if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { 
         glfwSetWindowShouldClose(window, true); 
   } 

   if (key == GLFW_KEY_UP && action == GLFW_PRESS) { 

         if (gameover) { 
               gameover = false; 
         } else { 

               if (grounded == true) { 

                     grounded = false; 

sphere->rigidBody->applyImpulse(btVector3(0.0f, 100.0f, 0.0f), 
   btVector3(0.0f, 0.0f, 0.0f)); 
                     printf("pressed up key \n"); 
               } 
         } 
   } 
}   
```

6.  Although we are calculating the score, the user still cannot see what the score is, so let's add text rendering to the game.

# Text rendering

For rendering text, we will use a library called FreeType, load in the font, and read the characters from it. FreeType can load a popular font format called TrueType. TrueType fonts have a `.ttf` extension.

TTFs contain vector information called glyphs that can be used to store any data. One use case is, of course, to represent characters with them.

So, when we want to render a particular glyph, we load the character glyph by specifying its size; the character will be generated without there being a loss in quality.

The source of the FreeType library can be downloaded from their website at [https://www.freetype.org/](https://www.freetype.org/) and the library can be built from it. The precompiled libraries can also be downloaded from [https://github.com/ubawurinna/freetype-windows-binaries](https://github.com/ubawurinna/freetype-windows-binaries).

Let's add the library to our project. Since we are developing for the 64-bit OS, we are interested in the `include` directory and the `win64` directory; they contain the `freetype.lib` and `freetype.dll` files for our version of the project:

1.  Create a folder called `freetype` in your dependencies folder and extract the files into it, as follows:

![](img/26c04855-bd22-4ccd-a0b7-9ba34233dcff.png)

2.  Open the project's properties and, under C/C++ in Additional Include Directory, add the `freetype` include directory location, as follows:

![](img/57339f31-2a05-43dc-aeaf-9155d6d2a478.png)

3.  Under Configuration Properties | Linker | General | Additional Library Directories, add the freetype `win64` directory, as follows:

![](img/c02482b4-37c3-4a6d-b849-a0bd9501f4dc.png)

4.  In the project directory, copy the `Freetype.dll` file from the `win64` directory and paste it here:

![](img/2c29c11e-af9c-4aa1-a429-2e7802f3d41f.png)

With the prep work out of the way, we can start working on the project.

5.  Create a class called `TextRenderer`, as well as a file called `TextRenderer.h` and a file called `TextRenderer.cpp`. We will add the functionality for text rendering to these files. In  `TextRenderer.h`, include the usual include headers for `GL` and `glm` as `b`, as follows:

```cpp
#include <GL/glew.h> 

#include "Dependencies/glm/glm/glm.hpp" 
#include "Dependencies/glm/glm/gtc/matrix_transform.hpp" 
#include "Dependencies/glm/glm/gtc/type_ptr.hpp"
```

6.  Next, we will include the headers for `freetype.h`, as follows:

```cpp
#include <ft2build.h> 
#include FT_FREETYPE_H    
```

7.  The `FT_FREETYPE_H` macro just includes `freetype.h` in the `freetype` directory. Then, we will `include <map>` as we will have to map each character's location, size, and other information. We will also `include <string>` and pass a string into the class to be rendered, as follows:

```cpp
#include <string> 
```

8.  For each glyph, we will need to keep track of certain properties. For this, we will create a `struct` called `Character`, as follows:

```cpp
struct Character { 
   GLuint     TextureID;  // Texture ID of each glyph texture 
   glm::ivec2 Size;       // glyph Size 
   glm::ivec2 Bearing;    // baseline to left/top of glyph 
   GLuint     Advance;    // id to next glyph 
}; 

```

For each glyph, we will store the texture ID of the texture we create for each character. We store the size of it, the bearing, which is the distance from the top left corner of the glyph from the baseline of the glyph, and the ID of the next glyph in the font file.

9.  This is what a font file looks like when it has all the character glyphs in it:

![](img/1f02801f-784a-4cca-9cb5-1af8ca0ac180.png)

Information regarding each character is stored in relation to the character adjacent to it, as follows:

![](img/c492e9ab-55b9-45ab-97b1-736e587acda3.png)

Each of these properties can be accessed on a per glyph basis after we load the font face of the `FT_Face` type. The width and height of each glyph can be accessed using the glyph property per font face, that is, `face->glyph as face->glyph->bitmap.width` and `face->glyph->bitmap.rows`.

The image data is available per glyph using the `bitmap.buffer` property, which we will be using when we create the texture for each glyph. The following code shows how all of this is implemented.

The next glyph in the font file can be accessed using the `advance.x` property of the glyph if the font is horizontally aligned.

That's enough theory about the library. If you are interested in finding out more, the necessary documentation is available on FreeType's website: [https://www.freetype.org/freetype2/docs/tutorial/step2.html#section-1](https://www.freetype.org/freetype2/docs/tutorial/step2.html#section-1).

Let's continue with the `TextRenderer.h` file and create the `TextRenderer` class, as follows:

```cpp
class TextRenderer{ 

public: 
   TextRenderer(std::string text, std::string font, int size, 
     glm::vec3 color, GLuint  program); 
   ~TextRenderer(); 

   void draw(); 
   void setPosition(glm::vec2 _position); 
   void setText(std::string _text); 

private: 
   std::string text; 
   GLfloat scale; 
   glm::vec3 color; 
   glm::vec2 position; 

   GLuint VAO, VBO, program; 
   std::map<GLchar, Character> Characters; 

};   
```

In the class under the public section, we add the constructor and destructor. In the constructor, we pass in the string we want to draw, the file we want to use, the size and color of the text we want to draw in, and pass in a shader program to use while drawing the font.

Then, we have the `draw` function to draw the text, a couple of setters to set the position, and a `setText` function to set a new string to draw if needed. In the private section, we have local variables for the text string, scale, color, and position. We also have member variables for `VAO`, `VBO`, and `program` so that we can draw the text string. At the end of the class, we create a map to store all the loaded characters and assign each `GLchar` to a character `struct` in the map. This is all we need to do for the `TextRenderer.h` file.

In the `TextRenderer.cpp` file, include the `TextRenderer.h` file at the top of the file and perform the following steps:

1.  Add the `TextRenderer` constructor implementation, as follows:

```cpp
TextRenderer::TextRenderer(std::string text, std::string font, int size, glm::vec3 color, GLuint program){ 

} 
```

In the constructor, we will add the functionally for loading all the characters and prep the class for drawing the text.

2.  Let's initialize the local variables, as follows:

```cpp
   this->text = text; 
   this->color = color; 
   this->scale = 1.0; 
   this->program = program; 
   this->setPosition(position); 
```

3.  Next, we need to set the projection matrix. For text, we specify the orthographic projection since it doesn't have any depth, as follows:

```cpp
   glm::mat4 projection = glm::ortho(0.0f, static_cast<GLfloat>
                         (800), 0.0f, static_cast<GLfloat>(600)); 
   glUseProgram(program); 
   glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 
      1, GL_FALSE, glm::value_ptr(projection)); 
```

The projection is created using the `glm::ortho` function, which takes origin x, window width, origin y, and window height as the parameters for creating the orthographic projection matrix. We will use the current program and pass the value for the projection matrix to a location called projection, and then pass this on to the shader. Since this value will never change, it is called and assigned once in the constructor.

4.  Before we load the font itself, we have to initialize the FreeType library, as follows:

```cpp
// FreeType 
FT_Library ft; 

// Initialise freetype 
if (FT_Init_FreeType(&ft)) 
std::cout << "ERROR::FREETYPE: Could not init FreeType Library" 
          << std::endl; 
```

5.  Now, we can load the font face itself, as follows:

```cpp
// Load font 
FT_Face face; 
if (FT_New_Face(ft, font.c_str(), 0, &face)) 
         std::cout << "ERROR::FREETYPE: Failed to load font" 
                   << std::endl; 

```

6.  Now, set the font size in pixels and disable the byte alignment restriction. If we don't restrict the byte alignment, the font will be drawn jumbled, so don't forget to add this:

```cpp
// Set size of glyphs 
FT_Set_Pixel_Sizes(face, 0, size); 

// Disable byte-alignment restriction 
glPixelStorei(GL_UNPACK_ALIGNMENT, 1); 
```

7.  Then, we will load the first `128` characters into the font we loaded and create and assign the texture ID, size, bearing, and advance. After, we will store the font in the characters map, as follows:

```cpp
   for (GLubyte i = 0; i < 128; i++){ 

         // Load character glyph  
         if (FT_Load_Char(face, i, FT_LOAD_RENDER)){ 
               std::cout << "ERROR::FREETYTPE: Failed to 
                            load Glyph" << std::endl; 
               continue; 
         } 

         // Generate texture 
         GLuint texture; 
         glGenTextures(1, &texture); 
         glBindTexture(GL_TEXTURE_2D, texture); 

         glTexImage2D( 
               GL_TEXTURE_2D, 
               0, 
               GL_RED, 
               face->glyph->bitmap.width, 
               face->glyph->bitmap.rows, 
               0, 
               GL_RED, 
               GL_UNSIGNED_BYTE, 
               face->glyph->bitmap.buffer 
               ); 

         // Set texture filtering options 
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 
         GL_CLAMP_TO_EDGE); 
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 
         GL_CLAMP_TO_EDGE); 
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
         GL_LINEAR); 
         glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
         GL_LINEAR); 

         // Create a character 
         Character character = { 
               texture, 
               glm::ivec2(face->glyph->bitmap.width, 
                           face->glyph->bitmap.rows), 
               glm::ivec2(face->glyph->bitmap_left, 
           face->glyph->bitmap_top), 
               face->glyph->advance.x 
         }; 

         // Store character in characters map 
         Characters.insert(std::pair<GLchar, Character>(i,
         character)); 
   } 
```

8.  Once the characters have been loaded, we can unbind the texture and destroy the font face and FreeType library, as follows:

```cpp
   glBindTexture(GL_TEXTURE_2D, 0); 

   // Destroy FreeType once we're finished 
   FT_Done_Face(face); 
   FT_Done_FreeType(ft);
```

9.  Each character will be drawn as a texture on a separate quad, so set the `VAO`/`VBO` for a quad, create a position attribute, and enable it, as follows:

```cpp
   glGenVertexArrays(1, &VAO); 
   glGenBuffers(1, &VBO); 

   glBindVertexArray(VAO); 

   glBindBuffer(GL_ARRAY_BUFFER, VBO); 
   glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, 
       GL_DYNAMIC_DRAW); 

   glEnableVertexAttribArray(0); 
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * 
      sizeof(GLfloat), 0); 

```

10.  Now, we need to unbind `VBO` and `VAO`, as follows:

```cpp
   glBindBuffer(GL_ARRAY_BUFFER, 0); 
   glBindVertexArray(0); 

```

That's all for the constructor. Now, we can move on to the draw function. Let's take a look:

1.  First, create the draw function's  implementation, as follows:

```cpp
void TextRenderer::draw(){
} 
```

2.  We will add the functionality for drawing to this function. First, we'll get the position where the text needs to start drawing, as follows:

```cpp
glm::vec2 textPos = this->position; 
```

3.  Then, we have to enable blending. If we don't enable blending, the whole quad for the text will be colored instead of just the area where the text is present, as shown in the image on the left:

![](img/4b5c4d13-726e-4ea5-b2cb-b6f3c6784d9b.png)

In the image on the left, where the S is supposed to be, we can see the whole quad colored in red, including the pixels where it is supposed to be transparent.

By enabling blending, we set the final color value as a pixel using the following equation:

*Color[final] = Color[Source] * Alpha[Source] + Color[Destination] * 1- Alpha[Source]*

Here, source color and source alpha are the color and alpha values of the text at a certain pixel location, while the destination color and alpha are the values of the color and alpha at the color buffer.

In this example, since we draw the text later, the destination color will be yellow, and the source color, which is the text, will be red. The destination alpha value is 1.0 while the yellow color is opaque. For the text, if we take a look at the S glyph, for example, within the S, which is the red area, it is opaque, but it is transparent.

Using this formula, let's calculate the final pixel color around the S where it is transparent using the following equation:

*Color[final] = (1.0f, 0.0f, 0.0f, 0.0f) * 0.0 + (1.0f, 1.0f, 0.0f, 1.0f) * (1.0f- 0.0f)
= (1.0f, 1.0f, 0.0f, 1.0f);*

This is just the yellow background color.

Conversely, within the S glyph, it is not transparent, so the alpha value is 1 at that pixel location. So, when we apply the same formula, we get the final color, as follows:

*Color[final] = (1.0f, 0.0f, 0.0f, 1.0f) * 1.0 + (1.0f, 1.0f, 0.0f, 1.0f) * (1.0f- 1.0f)*
*= (1.0f, 0.0f, 0.0f, 1.0f)*

This is just the red text color, as shown in the following diagram:

![](img/299dfb78-cf0f-4fac-97ca-908b9309fee7.png)

Let's see how this is implemented in practice.

4.  The `blend` function is as follows:

```cpp
   glEnable(GL_BLEND); 
```

Now, we need to set the source and destination blending factors, that is, `GL_SRC_ALPHA`. For the source pixel, we use its alpha value as-is, whereas, for the destination, we set the alpha to `GL_ONE_MINUS_SRC_ALPHA`, which is the source alpha minus one, as follows:

```cpp
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
```

By default, the value source and destination values are added. You can subtract, add, and divide as well.

5.  Now, we need to call the `glUSeProgram` function to set the program, set the text color to the uniform location, and set the default texture, as follows:

```cpp
   glUseProgram(program); 
   glUniform3f(glGetUniformLocation(program, "textColor"), 
      this->color.x, this->color.y, this->color.z); 
   glActiveTexture(GL_TEXTURE0); 
```

6.  Next, we need to bind the `VAO`, as follows:

```cpp
   glBindVertexArray(VAO); 
```

7.  Let's go through all the characters in the text we want to draw and get their size, the bearing, so that we can set the position, and the texture ID of each glyph we want to draw, as follows:

```cpp
   std::string::const_iterator c; 

   for (c = text.begin(); c != text.end(); c++){ 

         Character ch = Characters[*c]; 

         GLfloat xpos = textPos.x + ch.Bearing.x * this->scale; 
         GLfloat ypos = textPos.y - (ch.Size.y - ch.Bearing.y) * 
         this->scale; 

         GLfloat w = ch.Size.x * this->scale; 
         GLfloat h = ch.Size.y * this->scale; 

         // Per Character Update VBO 
         GLfloat vertices[6][4] = { 
               { xpos, ypos + h, 0.0, 0.0 }, 
               { xpos, ypos, 0.0, 1.0 }, 
               { xpos + w, ypos, 1.0, 1.0 }, 

               { xpos, ypos + h, 0.0, 0.0 }, 
               { xpos + w, ypos, 1.0, 1.0 }, 
               { xpos + w, ypos + h, 1.0, 0.0 } 
         }; 

         // Render glyph texture over quad 
         glBindTexture(GL_TEXTURE_2D, ch.TextureID); 

         // Update content of VBO memory 
         glBindBuffer(GL_ARRAY_BUFFER, VBO); 

         // Use glBufferSubData and not glBufferData 
         glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), 
         vertices);  

         glBindBuffer(GL_ARRAY_BUFFER, 0); 

         // Render quad 
         glDrawArrays(GL_TRIANGLES, 0, 6); 

         // Now advance cursors for next glyph (note that advance 
         is number of 1/64 pixels) 
         // Bitshift by 6 to get value in pixels (2^6 = 64 (divide 
         amount of 1/64th pixels by 64 to get amount of pixels)) 
         textPos.x += (ch.Advance >> 6) * this->scale;  
   } 
```

We will now bind the  `VBO` and pass in the vertex data for all the quads to be drawn using `glBufferSubData`. Once bound, the quads are drawn using `glDrawArrays` and we pass in `6` for the number of vertices to be drawn.

Then, we calculate `textPos.x`, which will determine where the next glyph will be drawn. We get this distance by multiplying the advance of the current glyph by the scale and adding it to the current text position's `x` component. A bit shift of `6` is done to `advance`, to get the value in pixels.

8.  At the end of the draw function, we unbind the vertex array and the texture, and then disable blending, as follows:

```cpp
glBindVertexArray(0); 
glBindTexture(GL_TEXTURE_2D, 0); 

glDisable(GL_BLEND);  
```

9.  Finally, we add the implementation of the `setPOsiton` and `setString` functions, as follows:

```cpp
void TextRenderer::setPosition(glm::vec2 _position){ 

   this->position = _position; 
} 

void TextRenderer::setText(std::string _text){ 
   this->text = _text; 
} 
```

We're finally done with the `TextRenderer` class. Now, let's learn how we can display the text in our game:

1.  In the `main.cpp` file, include`TextRenderer.h` at the top of the file and create a new object of the class called `label`, as follows:

```cpp
#include "TextRenderer.h" 

TextRenderer* label;  

```

2.  Create a new `GLuint` for the text shader program, as follows:

```cpp
GLuint textProgram 
```

3.  Then, create the new shaded program for the text, as follows:

```cpp
textProgram = shader.CreateProgram("Assets/Shaders/text.vs", "Assets/Shaders/text.fs"); 
```

4.  The `text.vs` and `text.fs` files are placed in the `Assets` directory under `Shaders.text.vs`, as follows:

```cpp
#version 450 core 
layout (location = 0) in vec4 vertex; 
uniform mat4 projection; 

out vec2 TexCoords; 

void main(){ 
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0); 
    TexCoords = vertex.zw; 
}   
```

We get the vertex position as an attribute and the projection matrix as a uniform. The texture coordinate is set in the main function and is sent out to the next shader stage. The position of the vertex of the quad is set by multiplying the local coordinates by the orthographic projection matrix in the `main()` function.

5.  Next, we'll move on to the fragment shader, as follows:

```cpp
#version 450 core 

in vec2 TexCoords; 

uniform sampler2D text; 
uniform vec3 textColor; 

out vec4 color; 

void main(){     
vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r); 
color = vec4(textColor, 1.0) * sampled; 
} 
```

We get the texture coordinate from the vertex shader and the texture and color as uniforms. A new out `vec4` is created called color to send out color information. In the `main()` function, we create a new `vec4` called sampled and store the r,g, and b values as `1`. We also store the red color as the alpha value to draw only the opaque part of the text. Then, a new `vec4` called color is created, in which the white color is replaced with the color we want the text to be drawn in, and we assign the color variable.

6.  Let's continue with the text label implementation. After the `addRigidBody` function in the `init` function, initialize the `label` object, as follows:

```cpp
label = new TextRenderer("Score: 0", "Assets/fonts/gooddog.ttf", 
        64, glm::vec3(1.0f, 0.0f, 0.0f), textProgram); 
   label->setPosition(glm::vec2(320.0f, 500.0f)); 

```

In the constructor, we set the string we want to render, pass in the location of the font file, and pass in the text height, the text color, and the text program. Then, we use the `setPosition` function to set the position of the text.

7.  Next, in the tick function, where we update the score, we update the text as well, as follows:

```cpp
         if (t.getOrigin().x() <= -18.0f) { 

               t.setOrigin(btVector3(18, 1, 0)); 
               score++; 
               label->setText("Score: " + std::to_string(score));
         }
```

8.  In the tick function, we reset the string when the game is over, as follows:

```cpp
               gameover = true; 
               score = 0; 
               label->setText("Score: " + std::to_string(score));
```

9.  In the `render` function, we call the `draw` function to draw the text, as follows:

```cpp
void renderScene(float dt){ 

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
   glClearColor(1.0, 1.0, 0.0, 1.0); 

   // Draw Objects 

   //light->draw(); 

   sphere->draw(); 
   enemy->draw(); 
   ground->draw(); 

   label->draw(); 
} 
```

Because of alpha blending, the text has to be drawn at the end, after all the other objects have been drawn.

10.  Finally, make sure the font file has been added to the `Assets` folder under `Fonts`, as follows:

![](img/3662e148-2020-4b4d-9b6f-31ff70aebe80.png)

A few font files have been provided that you can experiment with. More free fonts can be downloaded from [https://www.1001freefonts.com/](https://www.1001freefonts.com/) and [https://www.dafont.com/](https://www.dafont.com/). Build and run the game to see the text being drawn and updated:

![](img/72630a6f-c8f6-4ab4-be2e-90c46938acc1.png)

# Adding lighting

Finally, let's add some lighting to the objects in the scene, just to make the objects more interesting to look at. We'll do this by allowing the light renderer to be drawn in the scene. Here, the light is originating from the center of this sphere. Using the position of the light source, we will calculate whether a pixel is lit or not, as follows:

![](img/b87c3061-eb22-4f78-b700-678bff35778c.png)

The picture on the left shows the scene unlit. In contrast, the scene on the right is lit with the earth sphere and the ground is affected by the light source. The surface that is facing the light is brightest, for example, at the top of the sphere. This creates a **Specular** at the top of the sphere. Since the surface is farther from/at an angle to the light source, those pixel values slowly diffuse. Then, there are surfaces that are not facing the light source at all, such as the side of the ground facing us. However, they are still not completely black as they are still being lit by the light from the source, which bounces around and becomes part of the ambient light. **Ambient**, **Diffuse,** and **Specular** become major parts of the lighting model when we wish to light up an object. Lighting models are used to simulate lighting in computer graphics because, unlike the real world, we are limited by the processing power of our hardware.

The formula for the final color of the pixel according to the Phong shading model is as follows:

*C = ka* Lc+ Lc * max(0, n l) + ks * Lc * max(0, v r) p*

Here, we have the following attributes:

*   *k*[*a* ]is the ambient strength.
*   *L[c]* is the light color.
*   *n* is the surface normal.
*   *l* is the light direction.

*   *k[s]* is the specular strength.
*   *v* is the view direction.
*   *r* is the reflected light direction about the normal of the surface.
*   *p* is the Phong exponent, which will determine how shiny a surface is.

For the **n**, **l**, **v** and **r** vectors, refer to the following diagram:

![](img/d0e6daa6-231d-4983-91eb-50bad4fbdbf3.png)

Let's look at how to implement this in practice:

1.  All the lighting calculations are done in the fragment shader of the object, since this affects the final color of the object, depending on the light source and camera position. For each object to be lit, we also need to pass in the light color, diffuse, and specular strength. In the `MeshRenderer.h` file, change the constructor so that it takes the light source, diffuse, and specular strengths, as follows:

```cpp
MeshRenderer(MeshType modelType, std::string _name, Camera * 
   _camera, btRigidBody* _rigidBody, LightRenderer* _light, float 
   _specularStrength, float _ambientStrength);
```

2.  Include `lightRenderer.h` at the top of the file, as follows:

```cpp
#include "LightRenderer.h"
```

3.  In the private section of the class, add an object for `LightRenderer` and floats to store the ambient and specular `Strength`, as follows:

```cpp
        GLuint vao, vbo, ebo, texture, program; 
        LightRenderer* light;
        float ambientStrength, specularStrength;
```

4.  In the `MeshRenderer.cpp` file, change the implementation of the constructor and assign the variables that were passed into the local variables, as follows:

```cpp
MeshRenderer::MeshRenderer(MeshType modelType, std::string _name, 
   Camera* _camera, btRigidBody* _rigidBody, LightRenderer* _light, 
   float _specularStrength, float _ambientStrength) { 

   name = _name; 
   rigidBody = _rigidBody; 
   camera = _camera; 
   light = _light; 
   ambientStrength = _ambientStrength; 
   specularStrength = _specularStrength; 
... 
} 

```

5.  In the constructor, we also need to add a new normal attribute, as we will need the surface normal information for lighting calculations, as follows:

```cpp
glEnableVertexAttribArray(0); 
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
   (GLvoid*)0); 

 glEnableVertexAttribArray(1); 
 glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
      (void*)(offsetof(Vertex, Vertex::texCoords))); 
 glEnableVertexAttribArray(2);
 glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
     (void*)(offsetof(Vertex, Vertex::normal))); 

```

6.  In the `Draw` function, we pass the camera position, light position, light color, specular strength, and ambient strength as uniforms to the shader, as follows:

```cpp
   // Set Texture 
   glBindTexture(GL_TEXTURE_2D, texture); 

   // Set Lighting 
   GLuint cameraPosLoc = glGetUniformLocation(program, 
                         "cameraPos"); 
   glUniform3f(cameraPosLoc, camera->getCameraPosition().x,
   camera-> getCameraPosition().y, camera->getCameraPosition().z); 

   GLuint lightPosLoc = glGetUniformLocation(program, "lightPos"); 
   glUniform3f(lightPosLoc, this->light->getPosition().x, 
     this-> light->getPosition().y, this->light->getPosition().z); 

   GLuint lightColorLoc = glGetUniformLocation(program, 
                          "lightColor"); 
   glUniform3f(lightColorLoc, this->light->getColor().x, 
     this-> light->getColor().y, this->light->getColor().z); 

   GLuint specularStrengthLoc = glGetUniformLocation(program, 
                                "specularStrength"); 
   glUniform1f(specularStrengthLoc, specularStrength); 

   GLuint ambientStrengthLoc = glGetUniformLocation(
                               program, "ambientStrength"); 
   glUniform1f(ambientStrengthLoc, ambientStrength); 

   glBindVertexArray(vao);        
   glDrawElements(GL_TRIANGLES, indices.size(), 
      GL_UNSIGNED_INT, 0); 
   glBindVertexArray(0); 

```

7.  We also need to create new vertex and fragment shaders for the effect to take place. Let's create a new vertex shader called `LitTexturedModel.vs`, as follows:

```cpp
#version 450 core 
layout (location = 0) in vec3 position; 
layout (location = 1) in vec2 texCoord; 
layout (location = 2) in vec3 normal; 

out vec2 TexCoord; 
out vec3 Normal; 
out vec3 fragWorldPos; 

uniform mat4 vp; 
uniform mat4 model; 

void main(){ 

   gl_Position = vp * model *vec4(position, 1.0); 

   TexCoord = texCoord; 
   fragWorldPos = vec3(model * vec4(position, 1.0)); 
   Normal = mat3(transpose(inverse(model))) * normal; 

} 
```

8.  We add the new location layout in order to receive the normal attribute.

9.  Create a new out `vec3` so that we can send the normal information to the fragment shader. We will also create a new out `vec3` to send the world coordinates of a fragment. In the `main()` function, we calculate the world position of the fragment by multiplying the local position by the world matrix and store it in the `fragWorldPos` variable. The normal is also converted into world space. Unlike how we multiplied the local position, the model matrix that's used to convert into the normal world space needs to be treated differently. The normal is multiplied by the inverse of the model matrix and is stored in the normal variable. That's all for the vertex shader. Now, let's look at `LitTexturedModel.fs`.
10.  In the fragment shader, we get the texture coordinate, normal, and fragment world position. Next, we get the camera position, light position and color, specular and ambient strength uniforms, and the texture as uniform as well. The final pixel value will be stored in the out `vec4` called color, as follows:

```cpp
#version 450 core 

in vec2 TexCoord; 
in vec3 Normal; 
in vec3 fragWorldPos; 

uniform vec3 cameraPos; 
uniform vec3 lightPos; 
uniform vec3 lightColor; 

uniform float specularStrength; 
uniform float ambientStrength; 

// texture 
uniform sampler2D Texture; 

out vec4 color;    
```

11.  In the `main()` function of the shader, we add the lighting calculation, as shown in the following code:

```cpp
 void main(){ 

       vec3 norm = normalize(Normal); 
       vec4 objColor = texture(Texture, TexCoord); 

       //**ambient 
       vec3 ambient = ambientStrength * lightColor; 

       //**diffuse 
       vec3 lightDir = normalize(lightPos - fragWorldPos); 
       float diff = max(dot(norm, lightDir), 0.0); 
       vec3 diffuse = diff * lightColor; 

       //**specular  
       vec3 viewDir = normalize(cameraPos - fragWorldPos); 
       vec3 reflectionDir = reflect(-lightDir, norm); 
       float spec = pow(max(dot(viewDir, 
                    reflectionDir),0.0),128); 
       vec3 specular = specularStrength * spec * lightColor; 

       // lighting shading calculation 
       vec3 totalColor = (ambient + diffuse + specular) * 
       objColor.rgb; 

       color = vec4(totalColor, 1.0f); 

}  
```

12.  We get the normal and object color first. Then, as per the formula equation, we calculate the ambient part of the equation by multiplying the ambient strength and light color and store it in a `vec3` called ambient. For the diffuse part of the equation, we calculate the light direction from the position of the pixel in the world space by subtracting the two positions. The resulting vector is normalized and saved in `vec3 lightDir`. Then, we get the dot product of the normal and light directions.
13.  After this, we get the resultant value or `0`, whichever is bigger, and store it in a float called `diff`. This is multiplied by the light color and stored in `vec3` to get the diffuse color. For the specular part of the equation, we calculate the view direction by subtracting the camera position from the fragment world position.
14.  The resulting vector is normalized and stored in `vec3 specDir`. Then, the reflected light vector regarding the surface normal is calculated by using the reflect `glsl` intrinsic function and passing in the `viewDir` and surface normal.
15.  Then, the dot product of the view and reflected vector is calculated. The bigger value of the calculated value and `0` is chosen. The resulting float value is raised to the power of `128`. The value can be from *0 to 256*. The bigger the value, the shinier the object will appear. The specular value is calculated by multiplying the specular strength, the calculated spec value, and the light color stored in the specular `vec3`.

16.  Finally, the total shading is calculated by adding the three ambient, diffuse, and specular values together and then multiplying this by the object color. The object color is a `vec4`, so we convert it into a `vec3`. The total color is assigned to the color variable by converting `totalColor` into a `vec4`. To implement this in the project, create a new shader program called `litTexturedShaderProgram`, 
    as follows:

```cpp
GLuint litTexturedShaderProgram; 
Create the shader program and assign it to it in the init function in main.cpp. 
   litTexturedShaderProgram = shader.CreateProgram(
                              "Assets/Shaders/LitTexturedModel.vs",                 
                              "Assets/Shaders/LitTexturedModel.fs"); 
```

17.  In the add `rigidBody` function, change the shaders for the sphere, ground, and enemy, as follows:

```cpp
  // Sphere Rigid Body 

  btCollisionShape* sphereShape = new btSphereShape(1);
  btDefaultMotionState* sphereMotionState = new 
     btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), 
     btVector3(0, 0.5, 0)));

  btScalar mass = 13.0f;
  btVector3 sphereInertia(0, 0, 0);
  sphereShape->calculateLocalInertia(mass, sphereInertia);

  btRigidBody::btRigidBodyConstructionInfo 
     sphereRigidBodyCI(mass, sphereMotionState, sphereShape, 
     sphereInertia);

  btRigidBody* sphereRigidBody = new btRigidBody
                                 (sphereRigidBodyCI);

  sphereRigidBody->setFriction(1.0f);
  sphereRigidBody->setRestitution(0.0f);

  sphereRigidBody->setActivationState(DISABLE_DEACTIVATION);

  dynamicsWorld->addRigidBody(sphereRigidBody);

  // Sphere Mesh

  sphere = new MeshRenderer(MeshType::kSphere, “hero”, 
           camera, sphereRigidBody, light, 0.1f, 0.5f);
  sphere->setProgram(litTexturedShaderProgram);
  sphere->setTexture(sphereTexture);
  sphere->setScale(glm::vec3(1.0f));

  sphereRigidBody->setUserPointer(sphere);

  // Ground Rigid body

  btCollisionShape* groundShape = new btBoxShape(btVector3(4.0f,   
                                  0.5f, 4.0f));
  btDefaultMotionState* groundMotionState = new 
    btDefaultMotionState(btTransform(btQuaternion(0, 0, 0, 1), 
     btVector3(0, -1.0f, 0)));

  btRigidBody::btRigidBodyConstructionInfo 
    groundRigidBodyCI(0.0f, groundMotionState, groundShape, 
    btVector3(0, 0, 0));

  btRigidBody* groundRigidBody = new btRigidBody
                                 (groundRigidBodyCI);

  groundRigidBody->setFriction(1.0);
  groundRigidBody->setRestitution(0.0);

  groundRigidBody->setCollisionFlags(
     btCollisionObject::CF_STATIC_OBJECT);

  dynamicsWorld->addRigidBody(groundRigidBody);

  // Ground Mesh
  ground = new MeshRenderer(MeshType::kCube, “ground”,
           camera, groundRigidBody, light, 0.1f, 0.5f);
  ground->setProgram(litTexturedShaderProgram);
  ground->setTexture(groundTexture);
  ground->setScale(glm::vec3(4.0f, 0.5f, 4.0f));

  groundRigidBody->setUserPointer(ground);

  // Enemy Rigid body

  btCollisionShape* shape = new btBoxShape(btVector3(1.0f, 
                            1.0f, 1.0f));
  btDefaultMotionState* motionState = new btDefaultMotionState(
      btTransform(btQuaternion(0, 0, 0, 1), 
      btVector3(18.0, 1.0f, 0)));
  btRigidBody::btRigidBodyConstructionInfo rbCI(0.0f, 
     motionState, shape, btVector3(0.0f, 0.0f, 0.0f));

  btRigidBody* rb = new btRigidBody(rbCI);

  rb->setFriction(1.0);
  rb->setRestitution(0.0);

  //rb->setCollisionFlags(btCollisionObject::CF_KINEMATIC_OBJECT);

  rb->setCollisionFlags(btCollisionObject::CF_NO_CONTACT_RESPONSE);

  dynamicsWorld->addRigidBody(rb);

  // Enemy Mesh
  enemy = new MeshRenderer(MeshType::kCube, “enemy”, 
          camera, rb, light, 0.1f, 0.5f);
  enemy->setProgram(litTexturedShaderProgram);
  enemy->setTexture(groundTexture);
  enemy->setScale(glm::vec3(1.0f, 1.0f, 1.0f));

  rb->setUserPointer(enemy);

```

18.  Build and run the project to see the lighting shader take effect:

![](img/b06faca5-471a-44ba-ac28-74bdc467bf34.png)

As an exercise, try adding a texture to the background, just like we did in the SFML game.

# Summary

In this chapter, we saw how we can add collision detection between the game objects, and then we finished the game loop by adding controls and scoring. Using the font loading library FreeType, we loaded the TrueType font into our game to add scoring text to the game. Finally, to top it all off, we added lighting to the scene by adding the Phong lighting model to the objects.

There is still a lot that can be added graphically to add more realism to our game, such as framebuffers that add postprocessing effects. We could also add particle effects such as dust and rain. To find out more, I would highly recommend [learnopengl.com](http://learnopengl.com), as it is an amazing source if you wish to learn more about OpenGL.

In the next chapter, we will start exploring the Vulkan Rendering API and look at how it is different from OpenGL.