# Creating Your Game

In this chapter, we will make our project more flexible by adding game objects as classes instead of adding them to the `source.cpp` file. In this case, we will use classes to make the main character and the enemy. We will create a new rocket class that the player will be able to shoot at the enemy. We will then spawn enemies at regular intervals along with new rockets when we press a button. We will finally check for a collision between the rocket and the enemy and, accordingly, remove the enemy from the scene.

This chapter will cover the following topics:

*   Starting afresh
*   Creating the `Hero` class
*   Creating the `Enemy` class
*   Adding enemies
*   Creating the `Rocket` class
*   Adding rockets
*   Collision detection

# Starting afresh

Since we are going to create a new class for the main character, we will remove the code pertaining to the player character in the main file. Let's learn how to do this.

Remove all player-related code from the `main.cpp` file. After doing this, the file should appear as follows:

```cpp
#include "SFML-2.5.1\include\SFML\Graphics.hpp" 

sf::Vector2f viewSize(1024, 768); 
sf::VideoMode vm(viewSize.x, viewSize.y); 
sf::RenderWindow window(vm, "Hello SFML Game !!!", sf::Style::Default); 

sf::Vector2f playerPosition; 
bool playerMoving = false; 

sf::Texture skyTexture; 
sf::Sprite skySprite; 

sf::Texture bgTexture; 
sf::Sprite bgSprite; 

void init() { 

   skyTexture.loadFromFile("Assets/graphics/sky.png"); 
   skySprite.setTexture(skyTexture); 

   bgTexture.loadFromFile("Assets/graphics/bg.png"); 
   bgSprite.setTexture(bgTexture); 

} 

void updateInput() { 

   sf::Event event; 

   // while there are pending events... 
   while (window.pollEvent(event)) { 

      if (event.key.code == sf::Keyboard::Escape || event.type == 
          sf::Event::Closed) 
         window.close(); 

   } 

} 

void update(float dt) { 

} 

void draw() { 

   window.draw(skySprite); 
   window.draw(bgSprite); 

} 

int main() { 

   sf::Clock clock; 
   window.setFramerateLimit(60); 

   init(); 

   while (window.isOpen()) { 

      updateInput(); 

      sf::Time dt = clock.restart(); 
      update(dt.asSeconds()); 

      window.clear(sf::Color::Red); 

      draw(); 

      window.display(); 

   } 

   return 0; 
} 
```

# Creating the Hero class

We will now move on to create a new class by going through the following steps:

1.  Select the project in the solution explorer and then right-click and select Add | Class. In this class name, specify the name as `Hero`. You will see the `.h` and `.cpp` file sections automatically populated as `Hero.h` and `Hero.cpp` respectively. Click on Ok.

2.  In the `Hero.h` file, add the SFML graphics header and create the `Hero` class:

```cpp
#include "SFML-2.5.0\include\SFML\Graphics.hpp" 

class Hero{ 

}; 
```

3.  In the `Hero` class, we will create the methods and variables that will be required by the class. We will also create some public properties that will be accessible outside the class, as follows:

```cpp
public: 
   Hero(); 
   ~Hero(); 

   void init(std::string textureName, sf::Vector2f position, float 
   mass); 
   void update(float dt); 
   void jump(float velocity); 
   sf::Sprite getSprite(); 

```

Here, we have the constructor and destructor, which will be respectively called when an object is created and destroyed. We add an `init` function to pass a texture name, spawn the player, and specify a mass. We are specifying a mass here because we will be creating some very basic physics so that when we hit the jump button, the player will jump up and land back down on their feet.

Additionally, the `update`, `jump`, and `getSprite` functions will update the player position, make the player jump, and get the sprite of the player that is used for depicting the player character respectively.

4.  Apart from these `public` variables, we will also need some `private` variables that can only be accessed within the class. Add these in the `Hero` class, as follows:

```cpp
private: 

   sf::Texture m_texture; 
   sf::Sprite m_sprite; 
   sf::Vector2f m_position; 

int jumpCount = 0;    
float m_mass; 
   float m_velocity; 
   const float m_gravity = 9.80f; 
      bool m_grounded; 

```

In the `private` section, we create variables for `texture`, `sprite`, and `position` so that we can set these values locally. We have the `int` variable called `jumpCount` so that we can check the number of times the player character has jumped. This is needed because the player can sometimes double jump, which is something we don't want.

We will also need the `float` variables to store the player's mass, the velocity when they jump, and the gravitational force when they fall back to the ground, which is a constant. The `const` keyword tells the program that it is a constant and under no circumstance should the value be made changeable.

Lastly, we add a `bool` value to check whether the player is on the ground. Only when the player is on the ground can they start jumping.

5.  Next, in the `Hero.cpp` file, we will implement the functions that we added in the `.h` file. At the top of the `.cpp` file, include the `Hero.h` file and then add the constructor and destructor:

```cpp
#include "Hero.h" 

Hero::Hero(){ 

} 
```

The `::` symbol represents the scope resolution operator. Functions that have the same name can be defined in two different classes. In order to access the methods of a particular class, the scope resolution operator is used.

6.  Here, the `Hero` function is scoped to the `Hero` class:

```cpp
 Hero::~Hero(){ 

}
```

7.  Next, we will set up the `init` function, as follows:

```cpp
void Hero::init(std::string textureName, sf::Vector2f position, float mass){ 

   m_position = position; 
   m_mass = mass; 

   m_grounded = false; 

   // Load a Texture 
   m_texture.loadFromFile(textureName.c_str()); 

   // Create Sprite and Attach a Texture 
   m_sprite.setTexture(m_texture); 
   m_sprite.setPosition(m_position); 
   m_sprite.setOrigin(m_texture.getSize().x / 2, 
   m_texture.getSize().y / 2); 

} 
```

We set the position and mass to the local variable and set the grounded state to `false`. Then, we set the texture by calling `loadFromFile` and passing in the string of the texture name to it. The `c_str()` phrase returns a pointer to an array that contains a null-terminated sequence of characters (that is, a `C` string) representing the current value of the `string` object ([http://www.cplusplus.com/reference/string/string/c_str/](http://www.cplusplus.com/reference/string/string/c_str/)). We then set the sprite texture, position, and the origin of the sprite itself.

8.  Now, we add the `update` function where we implement the logic for updating the player position. The player's character cannot move left or right; instead, it can only move up, which is the *y* direction. When an initial velocity is applied, the player will jump up and then start falling down because of gravity. Add the `update` function to update the position of the hero, as follows:

```cpp
void Hero::update(float dt){ 

   m_force -= m_mass * m_gravity * dt; 

   m_position.y -= m_force * dt; 

   m_sprite.setPosition(m_position); 

   if (m_position.y >= 768 * 0.75f) { 

      m_position.y = 768 * 0.75f; 
      m_force = 0; 
      m_grounded = true; 
      jumpCount = 0; 
   } 

}  
```

When the velocity is applied to the character, the player will initially go up because of the force, but will then start coming down because of gravity. The resulting velocity acts in the downward direction, and is calculated by the following formula:

*Velocity = Acceleration × Time*

We multiply the acceleration by the mass so that the player falls faster. To calculate the distance moved vertically, we use the following formula:

*Distance = Velocity × Time*

Then, we calculate the distance moved between the previous and the current frame. We then set the position of the sprite based on the position that we calculated.

We also have a condition to check whether the player is at one-fourth of the distance from the bottom of the screen. We multiply this by `768`, which is the height of the window, and then multiply it by `.75f`, at which point the player is considered to be on the ground. If that condition is satisfied, we set the position of the player, set the resulting velocity to `0`, set the grounded Boolean to `true`, and, finally, reset the jump counter to `0`.

9.  When we want to make the player jump, we call the `jump` function, which takes an initial velocity. We will now add the `jump` function, as follows:

```cpp
void Hero::jump(float velocity){ 

   if (jumpCount < 2) { 
      jumpCount++; 

      m_velocity = VELOCITY; 
      m_grounded = false; 
   } 

}
```

Here, we first check whether `jumpCount` is less than `2` as we only want the player to jump twice. If `jumpCount` is less than `2`, then increase the `jumpCount` value by `1`, set the initial velocity, and set the grounded Boolean to `false`.

10.  Finally, we add the `getSprite` function, which simply gets the current sprite, as follows:

```cpp
 sf::Sprite Hero::getSprite(){ 

   return m_sprite; 
}  
```

Congrats! We now have our `Hero` class ready. Let's use it in the `source.cpp` file by going through the following steps:

1.  Include `Hero.h` at the top of the `main.cpp` file:

```cpp
#include "SFML-2.5.1\include\SFML\Graphics.hpp" 
#include "Hero.h"
```

2.  Next, add an instance of the `Hero` class, as follows:

```cpp
sf::Texture skyTexture; 
sf::Sprite skySprite; 

sf::Texture bgTexture; 
sf::Sprite bgSprite; 
Hero hero;
```

3.  In the `init` function, initialize the `Hero` class:

```cpp
   // Load bg Texture 

   bgTexture.loadFromFile("Assets/graphics/bg.png"); 

   // Create Sprite and Attach a Texture 
   bgSprite.setTexture(bgTexture); 

   hero.init("Assets/graphics/hero.png", sf::Vector2f(viewSize.x *
 0.25f, viewSize.y * 0.5f), 200); 
```

Here, we set the texture picture; to do so, set the position to be at `.25` (or 25%) from the left of the screen and center it along the `y` axis. We also set the mass as `200`, as our character is quite chubby.

4.  Next, we want the player to jump when we press the up arrow key. Therefore, in the `updateInput` function, while polling for window events, we add the following code:

```cpp
while (window.pollEvent(event)) {  
    if (event.type == sf::Event::KeyPressed) {
 if (event.key.code == sf::Keyboard::Up) {
 hero.jump(750.0f);
 }
 }
      if (event.key.code == sf::Keyboard::Escape || event.type == 
       sf::Event::Closed) 
         window.close(); 

   }  
```

Here, we check whether a key was pressed by the player. If a key is pressed and the button is the up arrow on the keyboard, then we call the `hero.jump` function and pass in an initial velocity value of `750`.

5.  Next, in the `update` function, we call the `hero.update` function, as follows:

```cpp
void update(float dt) { 
 hero.update(dt); 
} 

```

6.  Finally, in the `draw` function, we draw the hero sprite:

```cpp
void draw() { 

   window.draw(skySprite); 
   window.draw(bgSprite); 
 window.draw(hero.getSprite());

}
```

7.  You can now run the game. When the player is on the ground, press the up arrow button on the keyboard to see the player jump. When the player is in the air, press the jump button again and you will see the player jump again in midair:

![](img/af75e5e7-b61e-49ac-ba1c-7c5fdde8c175.png)

# Creating the Enemy class

The player character looks very lonely. She is ready to cause some mayhem, but there is nothing to shoot right now. Let's add some enemies to solve this problem:

1.  The enemies will be created using an enemy class; let's create a new class and call it `Enemy`.
2.  Just like the `Hero` class, the `Enemy` class will have a `.h` file and a `.cpp` file. In the `Enemy.h` file, add the following code:

```cpp
#pragma once 
#include "SFML-2.5.1\include\SFML\Graphics.hpp" 

class Enemy 
{ 
public: 
   Enemy(); 
   ~Enemy(); 

   void init(std::string textureName, sf::Vector2f position, 
     float_speed); 
   void update(float dt); 
   sf::Sprite getSprite(); 

private: 

   sf::Texture m_texture; 
   sf::Sprite m_sprite; 
   sf::Vector2f m_position; 
   float m_speed; 

}; 
```

Here, the `Enemy` class, just like the `Hero` class, also has a constructor and destructor. Additionally, it has an `init` function that takes in the texture and position; however, instead of mass, it takes in a float variable that will be used to set the initial velocity of the enemy. The enemy won't be affected by gravity and will only spawn from the right of the screen and move toward the left of the screen. There are also `update` and `getSprite` functions; since the enemy won't be jumping, there won't be a `jump` function. Lastly, in the private section, we create local variables for texture, sprite, position, and speed.

3.  In the `Enemy.cpp` file, we add the constructor, destructor, `init`, `update`, and `getSprite` functions:

```cpp
#include "Enemy.h" 

Enemy::Enemy(){} 

Enemy::~Enemy(){} 

void Enemy::init(std::string textureName, sf::Vector2f position, 
    float _speed) { 

   m_speed = _speed; 
   m_position = position; 

   // Load a Texture 
   m_texture.loadFromFile(textureName.c_str()); 

   // Create Sprite and Attach a Texture 
   m_sprite.setTexture(m_texture); 
   m_sprite.setPosition(m_position); 
   m_sprite.setOrigin(m_texture.getSize().x / 2,    
   m_texture.getSize().y / 2); 

} 
```

Don't forget to include `Enemy.h` at the top of the main function. We then add the constructor and destructor. In the `init` function, we set the local speed and position values. Next, we load `Texture` from the file and set the texture, position, and origin of the enemy sprite.

4.  In the `update` and `getSprite` functions, we update the position and get the enemy sprite:

```cpp
 void Enemy::update(float dt) { 

   m_sprite.move(m_speed * dt, 0); 

} 

sf::Sprite Enemy::getSprite() { 

   return m_sprite; 
}
```

5.  We have our `Enemy` class ready. Let's now see how we can use it in the game.

# Adding enemies

In the `main.cpp` class, include the `Enemy` header file. Since we want more than one enemy instance, we need to add a vector called `enemies` and then add all the newly-created enemies to it.

In the context of the following code, the `vector` phrase has absolutely nothing to do with math, but rather with a list of objects. In fact, it is like an array in which we can store multiple objects. Vectors are used instead of an array because vectors are dynamic in nature, so it makes it easier to add and remove objects from the list (unlike an array, which, by comparison, is a static list). Let's get started:

1.  We need to include `<vector>` in the `main.cpp` file, as follows:

```cpp
#include "SFML-2.5.1\include\SFML\Graphics.hpp" 
#include <vector> 

#include "Hero.h" 
#include "Enemy.h" 
```

2.  Next, add a new variable called `enemies` of the `vector` type, which will store the `Enemy` data type in it:

```cpp
sf::Texture bgTexture; 
sf::Sprite bgSprite; 

Hero hero; 

std::vector<Enemy*> enemies;  
```

3.  In order to create a vector of a certain object type, you use the `vector` keyword, and inside the arrow brackets, specify the data type that the vector will hold, and then specify a name for the vector you have created. In this way, we can create a new function called `spawnEnemy()` and add a prototype for it at the top of the main function.

When any function is written below the main function, the main function will not be aware that such a function exists. Therefore, a prototype will be created and placed above the main function. This means that the function can now be implemented below the main function—essentially, the prototype tells the main function that there is a function that will be implemented below it, and so to keep a lookout for it.

```cpp
sf::Vector2f viewSize(1024, 768);
sf::VideoMode vm(viewSize.x, viewSize.y); 
sf::RenderWindow window(vm, "Hello SFML Game !!!", sf::Style::Default); 

void spawnEnemy(); 
```

Now, we want the enemy to spawn from the right of the screen, but we also want the enemy to spawn at either the same height as the player, slightly higher than the player, or much higher than the player, so that the player will have to use a single jump or a double jump to attack the enemy.

4.  To do this, we add some randomness to the game to make it less predictable. For this, we add the following line of code underneath the `init` function:

```cpp
hero.init("Assets/graphics/hero.png", sf::Vector2f(viewSize.x * 
0.25f, viewSize.y * 0.5f), 200); 

srand((int)time(0)); 
```

The `srand` phrase is a pseudorandom number that is initialized by passing a seed value. In this case, we are passing in the current time as a seed value.

For each seed value, a series of numbers will be generated. If the seed value is always the same, then the same series of numbers will be generated. That is why we pass in the time value—so that the sequence of numbers that is generated will be different each time. We can get the next random number in the series by calling the `rand` function.

5.  Next, we add the `spawnEnemy` function, as follows:

```cpp
void spawnEnemy() { 

   int randLoc = rand() % 3; 

   sf::Vector2f enemyPos; 

   float speed; 

   switch (randLoc) { 

   case 0: enemyPos = sf::Vector2f(viewSize.x, viewSize.y * 0.75f);
   speed = -400; break; 

   case 1: enemyPos = sf::Vector2f(viewSize.x, viewSize.y * 0.60f); 
   speed = -550; break; 

   case 2: enemyPos = sf::Vector2f(viewSize.x, viewSize.y * 0.40f); 
   speed = -650; break; 

   default: printf("incorrect y value \n"); return; 

   } 

   Enemy* enemy = new Enemy(); 
   enemy->init("Assets/graphics/enemy.png", enemyPos, speed); 

   enemies.push_back(enemy); 
} 
```

Here, we first get a random number—this will create a new random number from `0` to `2` because of the modulo operator while getting the random location. So the value of `randLoc` will either be `0`, `1`, or `2` each time the function is called.

A new `enemyPos` variable is created that will be assigned depending upon the `randLoc` value. We will also set the speed of the enemy depending on the `randLoc` value; for this, we create a new float called `speed`, which we will assign later. We then create a `switch` statement that takes in the `randLoc` value—this enables the random location to spawn the enemy.

Depending upon the scenario, we can set the `enemyPosition` variable and the `speed` of the enemy:

*   When `randLoc` is `0`, the enemy spawns from the bottom and moves with a speed of `-400`.
*   When `randLoc` is `1`, the enemy spawns from the middle of the screen and moves at a speed of `-500`.
*   When the value of `randLoc` is `2`, then the enemy spawns from the top of the screen and moves at the faster speed of `-650`.
*   If `randLoc` is none of these values, then a message is printed out saying that the value of *y* is incorrect, and instead of breaking, we return to make sure that the enemy doesn't spawn in a random location.

To print out the message to the console, we can use the `printf` function, which takes a string value. At the end of the string, we specify `\n`; this is a keyword to tell the compiler that it is the end of the line, and whatever is written after needs to be put in a new line, similar to when calling `std::cout`.

6.  Once we know the position and speed, we can then create the enemy object itself and initialize it. Note that the enemy is created as a pointer; otherwise, the reference to the texture is lost and the texture won't display when the enemy is spawned. Additionally, when we create the enemy as a raw pointer with a new keyword, the system allocates memory, which we will have to delete later.
7.  After the enemy is created, we add it to the `enemies` vector by calling the `push` function of the vectors.
8.  We want the enemy to spawn automatically at regular intervals. For this, we create two new variables to keep track of the current time and spawn a new enemy every `1.125` seconds.
9.  Next, create two new variables of the `float` type, called `currentTime` and `prevTime`:

```cpp
Hero hero; 

std::vector<Enemy*> enemies; 

float currentTime; 
float prevTime = 0.0f;  
```

10.  Then, in the `update` function, after updating the `hero` function, add the following lines of code in order to create a new enemy:

```cpp
 hero.update(dt); 
 currentTime += dt;
 // Spawn Enemies
 if (currentTime >= prevTime + 1.125f)))) {
 spawnEnemy();
 prevTime = currentTime;
}
```

First, we increment the `currentTime` variable. This variable will begin increasing as soon as the game starts so that we can track how long it has been since we started the game. Next, we check whether the current time is greater than or equal to the previous time plus `1.125` seconds, as that is when we want the new enemy to spawn. If it is `true`, then we call the `spawnEnemy` function, which will create a new enemy. We also set the previous time as equal to the current time so that we know when the last enemy was spawned. Good! So, now that we have enemies spawning in the game, we can `update` the enemies and also `draw` them.

11.  In the `update` function, we also create a `for` loop to update the enemies and delete the enemies once they go beyond the left of the screen. To do this, we add the following code to the `update` function:

```cpp
   // Update Enemies 

   for (int i = 0; i < enemies.size(); i++) { 

      Enemy *enemy = enemies[i]; 

      enemy->update(dt); 

      if (enemy->getSprite().getPosition().x < 0) { 

         enemies.erase(enemies.begin() + i); 
         delete(enemy); 

      } 
   } 
```

This is where the use of vectors is really helpful. With vectors, we are able to add, delete, and insert elements in the vector. In the example here, we get the reference of the enemy at the location index of `i` in the vector. If that enemy goes offscreen and needs to be deleted, then we can just use the `erase` function and pass the location index from the beginning of the vector to remove the enemy at that index. When we reset the game, we also delete the local reference of the enemy we created. This will also free the memory space that was allocated when we created the new enemy.

12.  In the `draw` function, we go through each of the enemies in a `for...each` loop and draw them:

```cpp
window.draw(skySprite); 
window.draw(bgSprite); 

window.draw(hero.getSprite()); 

for (Enemy *enemy : enemies) { 
  window.draw(enemy->getSprite()); 
}
```

We use the `for...each` loop to go through all the enemies, since the `getSprite` function needs to be called on all of them. Interestingly, we didn't use `for...each` when we had to update the enemies because with the `for` loop, we can simply use the index of the enemy if we have to delete it.

13.  Finally, add the `Enemy.png` file to the `Assets/graphics` folder. Now, when you run the game, you will see enemies spawning at different heights and moving toward the left of the screen:

![](img/8d000619-b80f-4cdd-9aa3-7c05f5092e44.png)

# Creating the Rocket class

The game has enemies in it now, but the player still can't shoot at them. Let's create some rockets so that these can be launched from the player's bazooka by going through the following steps:

1.  In the project, create a new class called `Rocket`. As you can see from the following code block, the `Rocket.h` class is very similar to the `Enemy.h` class:

```cpp
#pragma once 

#include "SFML-2.5.1\include\SFML\Graphics.hpp" 

class Rocket 
{ 
public: 
   Rocket(); 
   ~Rocket(); 

   void init(std::string textureName, sf::Vector2f position, 
      float_speed); 
   void update(float dt); 
   sf::Sprite getSprite(); 

private: 

   sf::Texture m_texture; 
   sf::Sprite m_sprite; 
   sf::Vector2f m_position; 
   float m_speed; 

}; 
```

The `public` section contains the `init`, `update`, and `getSprite` functions. The `init` function takes in the name of the texture to load, the position to set, and the speed at which the object is initialized. The `private` section has local variables for the `texture`, `sprite`, `position`, and `speed`.

2.  In the `Rocket.cpp` file, we add the constructor and destructor, as follows:

```cpp
#include "Rocket.h" 

Rocket::Rocket(){ 
} 

Rocket::~Rocket(){ 
} 
```

In the `init` function, we set the sp`e`ed and `position` variables. Then, we set the `texture` variable and initialize the sprite with the `texture` variable.

3.  Next, we set the `position` variable and origin of the sprite, as follows:

```cpp
void Rocket::init(std::string textureName, sf::Vector2f position, float _speed){ 

   m_speed = _speed; 
   m_position = position; 

   // Load a Texture 
   m_texture.loadFromFile(textureName.c_str()); 

   // Create Sprite and Attach a Texture 
   m_sprite.setTexture(m_texture); 
   m_sprite.setPosition(m_position); 
   m_sprite.setOrigin(m_texture.getSize().x / 2, 
     m_texture.getSize().y / 2); 

} 
```

4.  In the `update` function, the object is moved according to the `speed` variable:

```cpp
void Rocket::update(float dt){ 
   \ 
   m_sprite.move(m_speed * dt, 0); 

} 
```

5.  The `getSprite` function returns the current sprite, as follows:

```cpp
sf::Sprite Rocket::getSprite() { 

   return m_sprite; 
} 
```

# Adding rockets

Now that we have created the rockets, let's learn how to add them:

1.  In the `main.cpp` file, we include the `Rocket.h` class as follows:

```cpp
#include "Hero.h" 
#include "Enemy.h" 
#include "Rocket.h" 
```

2.  We then create a new vector of `Rocket` called `rockets`, which takes in `Rocket`:

```cpp
std::vector<Enemy*> enemies; 
std::vector<Rocket*> rockets;  
```

3.  In the `update` function, after we have updated all the enemies, we update all the rockets. We also delete the rockets that go beyond the right of the screen:

```cpp
   // Update Enemies 

   for (int i = 0; i < enemies.size(); i++) { 

      Enemy* enemy = enemies[i]; 

      enemy->update(dt); 

      if (enemy->getSprite().getPosition().x < 0) { 

         enemies.erase(enemies.begin() + i); 
         delete(enemy); 

      } 
   } 
 // Update rockets

 for (int i = 0; i < rockets.size(); i++) {

 Rocket* rocket = rockets[i];

 rocket->update(dt);

 if (rocket->getSprite().getPosition().x > viewSize.x) {
 rockets.erase(rockets.begin() + i);
 delete(rocket);
 }
}
```

4.  Finally, we draw all the rockets with the `draw` function by going through each rocket in the scene:

```cpp
    for (Enemy *enemy : enemies) { 

      window.draw(enemy->getSprite()); 
   } 

for (Rocket *rocket : rockets) {
 window.draw(rocket->getSprite());
}
```

5.  Now, we can actually shoot the rockets. In the `main.cpp`, class, create a new function called `shoot()` and add a prototype for it at the top of the main function.

```cpp
void spawnEnemy(); 
void shoot(); 
```

6.  In the `shoot` function, we will add the functionality to shoot the rockets. We will spawn new rockets and push them back to the `rockets` vector. You can add the `shoot` function as follows:

```cpp
void shoot() { 

   Rocket* rocket = new Rocket(); 

rocket->init("Assets/graphics/rocket.png",  
            hero.getSprite().getPosition(),  
    400.0f); 

   rockets.push_back(rocket); 

} 
```

When this function is called, it creates a new `Rocket` and initializes it with the `Rocket.png` file, sets the position of it as equal to the position of the hero sprite, and then sets the velocity to `400.0f`. The rocket is then added to the `rockets` vector.

7.  Now, in the `updateInput` function, add the following code so that when the down arrow key is pressed, the `shoot` function is called:

```cpp
   if (event.type == sf::Event::KeyPressed) { 

      if (event.key.code == sf::Keyboard::Up) { 

         hero.jump(750.0f); 
      } 

      if (event.key.code == sf::Keyboard::Down) { 

         shoot(); 
      } 
   }  
```

8.  Don't forget to place the `rocket.png` file in the `assets` folder. Now, when you run the game and press the down arrow key, a rocket is fired:

![](img/f55476be-9e97-4fe2-9e6c-c0d5afdd0fc9.png)

# Collision detection

For the final section of this chapter, let's add some collision detection so that the rocket actually kills an enemy when they `both` come into contact with each other:

1.  Create a new function called `checkCollision`, and then create a prototype for it at the top of the main function:

```cpp
void spawnEnemy(); 
void shoot(); 

bool checkCollision(sf::Sprite sprite1, sf::Sprite sprite2); 
```

2.  This function takes two sprites so that we can check the intersection of one with the other. Add the following code for the function in the same place that we added the `shoot` function:

```cpp
void shoot() { 

   Rocket* rocket = new Rocket(); 

   rocket->init("Assets/graphics/rocket.png", 
     hero.getSprite().getPosition(), 400.0f); 

   rockets.push_back(rocket); 

} 

bool checkCollision(sf::Sprite sprite1, sf::Sprite sprite2) { 

   sf::FloatRect shape1 = sprite1.getGlobalBounds(); 
   sf::FloatRect shape2 = sprite2.getGlobalBounds(); 

   if (shape1.intersects(shape2)) { 

      return true; 

   } 
   else { 

      return false; 

   } 

}
```

Inside this `checkCollision` function, we create two local variables of the `FloatRect` type. We then assign the `GlobalBounds` of the sprites to each `FloatRect` variable named `shape1` and `shape2`. The `GlobalBounds` gets the rectangular region of the sprite that the object is spanning from where it is currently.

The `FloatRect` type is simply a rectangle; we can use the `intersects` function to check whether this rectangle intersects with another rectangle. If the first rectangle intersects with another rectangle, then we return `true` to say that there is an intersection or collision between the sprites. If there is no intersection, then we return `false`.

3.  In the `update` function, after updating the `enemy` and `rocket` classes, we check the collision between each rocket and each enemy in a nested `for` loop. You can add the collision check as follows:

```cpp
   // Update rockets 

   for (int i = 0; i < rockets.size(); i++) { 

      Rocket* rocket = rockets[i]; 

      rocket->update(dt); 

      if (rocket->getSprite().getPosition().x > viewSize.x) { 

         rockets.erase(rockets.begin() + i); 
         delete(rocket); 

      } 

   } 

    // Check collision between Rocket and Enemies 

   for (int i = 0; i < rockets.size(); i++) { 
      for (int j = 0; j < enemies.size(); j++) { 

         Rocket* rocket = rockets[i]; 
         Enemy* enemy = enemies[j]; 

         if (checkCollision(rocket->getSprite(), 
            enemy->getSprite())) { 

            rockets.erase(rockets.begin() + i); 
            enemies.erase(enemies.begin() + j); 

            delete(rocket); 
            delete(enemy); 

            printf(" rocket intersects enemy \n"); 
         } 

      } 
   }   
```

Here, we create a double `for` loop, call the `checkCollision` function, and then pass each rocket and enemy into it to check the intersection between them.

4.  If there is an intersection, we remove the rocket and enemy from the vector and delete them from the scene. With this, we are done with collision detection.

# Summary

In this chapter, we created a separate `Hero` class so that all the code pertaining to the `Hero` class was in one single file. In this `Hero` class, we managed jumping and the shooting of the rockets in the class. Next, we created the `Enemy` class, because for every hero, there needs to be a villain in the story! We learned how to add enemies to a vector so that it is easier to loop between the enemies in order to update their position. We also created a `Rocket` class and managed the rockets using a vector. Finally, we learned how to check for collisions between the enemies and the rockets. This creates the foundation of the gameplay loop.

In the next chapter, we will finish the game, adding sound and text to it in order to give audio feedback to the player and show the current score.