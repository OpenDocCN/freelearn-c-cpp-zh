# Finalizing Your Game

In the previous chapter, we looked at how to create the game; in this chapter, we will finish the **Gameloop** so that you can play the game. The objective of the game is to make sure that none of the enemies are able to make it to the left of the screen. If they do, it is game over.

We will add a scoring system so that the player knows how much they have scored in a round. For each enemy that is shot down, the player will get one point. We will also add text to the game in order to display the title of the game, the player's score, and a small tutorial that shows you how to play the game.

At the end of this chapter, we will embellish the game. We will add audio that will be used as background music, as well as sound effects for when the player shoots the rocket and when the player's rockets hit the enemy. We will also add some animation to the player so that the character looks more lively.

In this chapter, we will cover the following topics:

*   Finishing the Gameloop and adding scoring
*   Adding text
*   Adding audio
*   Adding player animations 

So, let's begin!

# Finishing the Gameloop and adding scoring

The following steps will show you how to finish the Gameloop and add scoring to the game code:

1.  Add two new variables to the `source.cpp` file: one of the `int` type, called `score`, and one of the `bool` type, called `gameover`. Initialize the score to `0` and `gameover` to `true`:

```cpp
std::vector<Enemy*> enemies; 
std::vector<Rocket*> rockets; 

float currentTime; 
float prevTime = 0.0f; 

int score = 0; 
bool gameover = true; 
```

2.  Create a new function called `reset()`. We will use this to reset the variables. Create a prototype for the reset function at the top of the `source.cpp` file:

```cpp
bool checkCollision(sf::Sprite sprite1, sf::Sprite sprite2); 
void reset(); 
```

At the bottom of the `source.cpp` file, after where we created the `checkCollision` function, add the reset function itself so that when the game resets, all the values are also reset. To do this, use the following code:

```cpp
void reset() { 

   score = 0; 
   currentTime = 0.0f; 
   prevTime = 0.0; 

   for (Enemy *enemy : enemies) { 
         delete(enemy); 
   } 
   for (Rocket *rocket : rockets) { 
         delete(rocket); 
   } 

   enemies.clear(); 
   rockets.clear(); 
}
```

If the game is over, pressing the down arrow key once will restart the game. Once the game starts again, the `reset()` function will be called. In the `reset()` function, we need to set `score`, `currentTime`, and `prevTime` to `0`. 

When the game resets, remove any instantiated enemy and rocket objects by deleting and thus freeing the memory. This also clears the vectors that were holding a reference to the now-deleted objects. Now that we've set up the variables and the reset function, let's use them in the game to reset the values when we restart the game.

In the `UpdateInput` function, in the `while` loop, where we check whether the down arrow key on the keyboard was pressed, we will add an `if` condition to check whether the game is over. If it is over, we'll set the `gameover` bool to `false` so that the game is ready to start, and we'll reset the variables by calling the `reset` function, as follows:

```cpp
           if (event.key.code == sf::Keyboard::Down) { 

             if (gameover) {
                gameover = false;
                reset();
              } 
            else { 
                 shoot(); 
                  }                
            } 
```

Here, `shoot()` is moved into an `else` statement so that the player can only shoot if the game is running.

Next, we will set the `gameover` condition to `true` when an enemy goes beyond the left-hand side of the screen.

When we update the enemies, the enemy will be deleted when it disappears from the screen, and we will also set the `gameover` condition to `true`.

Add the following code for updating the enemies to the `update()` function:

```cpp
// Update Enemies 
   for (int i = 0; i < enemies.size(); i++) { 

         Enemy* enemy = enemies[i]; 

         enemy->update(dt); 

         if (enemy->getSprite().getPosition().x < 0) { 

               enemies.erase(enemies.begin() + i); 
               delete(enemy); 
               gameover = true;
         } 
   } 
```

3.  Here, we want to update the game if `gameover` is `false`. In the `main` function, before we update the game, we will add a check to find out whether the game is over. If the game is over, we will not update the game. To do this, use the following code:

```cpp
   while (window.isOpen()) { 

         ////update input 
         updateInput(); 

         //// +++ Update Game Here +++ 
         sf::Time dt = clock.restart(); 
         if(!gameover)
             update(dt.asSeconds());
         //// +++ Draw Game Here ++ 

         window.clear(sf::Color::Red); 

         draw(); 

         // Show everything we just drew 
         window.display(); 

   } 
```

We will also increase the score when the rocket collides with an enemy. This means that, in the `update()` function, when we delete the rocket and enemy after the intersection, we will also update the score:

```cpp
   // Check collision between Rocket and Enemies 

   for (int i = 0; i < rockets.size(); i++) { 
         for (int j = 0; j < enemies.size(); j++) { 

               Rocket* rocket = rockets[i]; 
               Enemy* enemy = enemies[j]; 

               if (checkCollision(rocket->getSprite(), enemy-
                   >getSprite())) { 

                     score++; 

                     rockets.erase(rockets.begin() + i); 
                     enemies.erase(enemies.begin() + j); 

                     delete(rocket); 
                     delete(enemy); 

                     printf(" rocket intersects enemy \n"); 
               } 

         } 
   } 
```

When you run the game, start the game by pressing the down arrow key. When one of the enemies goes past the left-side of the screen, the game will end. When you press the down arrow key again, the game will restart.

The gameloop is now complete, but we still can't see the score. To do this, let's add some text to the game.

# Adding text

These steps will guide you through how to add text to the game:

1.  Create an `sf::Font` called `headingFont` so that we can load the font and then use it to display the name of the game. At the top of the screen, where we created all the variables, create the `headingFont` variable, as follows:

```cpp
int score = 0; 
bool gameover = true; 

// Text 
sf::Font headingFont;   
```

2.  In the `init()` function, right after we loaded `bgSprite`, load the font using the `loadFromFile` function:

```cpp
   // Create Sprite and Attach a Texture 
   bgSprite.setTexture(bgTexture); 

   // Load font 

   headingFont.loadFromFile("Assets/fonts/SnackerComic.ttf"); 
```

Since we will need a font to be loaded in from the system, we have to place the font in the `fonts` directory, which can be found under the `Assets` directory. Make sure you place the font file there. The font we will be using for the heading is the `SnackerComic.ttf` file. I have also included the `arial.ttf` file, which we will use to display the score, so make sure you add that as well.

3.  Create the `headingText` variable using the `sf::Text` type so that we can display the heading of the game. Do this at the start of the code:

```cpp
sf::Text headingText;   
```

4.  In the `init()` function, after loading `headingFont`, we will add the code to create the heading for the game:

```cpp
   // Set Heading Text 
   headingText.setFont(headingFont); 
   headingText.setString("Tiny Bazooka"); 
   headingText.setCharacterSize(84); 
   headingText.setFillColor(sf::Color::Red); 
```

We need to set the font for the heading text using the `setFont` function. In `setFont`, pass the `headingFont` variable that we just created.

We need to tell `headingText` what needs to be displayed. For that, we will use the `setString` function and pass in the `TinyBazooka` string since that is the name of the game we just made. Pretty cool name, huh?

Let's set the size of the font itself. To do this, we will use the `setCharacterSize` function and pass in `84` as the size in pixels so that it is clearly visible. Now, we can set the color to red using the `setFillColor` function.

5.  We want the heading to be centered on the viewport, so we will get the bounds of the text and set its origin to the `center` of the viewport in the *x* and *y* directions. Set the position of the text so that it's at the center of the x-direction and `0.10` of the height from the top along the *y-*direction:

```cpp
   sf::FloatRect headingbounds = headingText.getLocalBounds(); 
   headingText.setOrigin(headingbounds.width/2, 
      headingbounds.height / 2); 
   headingText.setPosition(sf::Vector2f(viewSize.x * 0.5f, 
      viewSize.y * 0.10f));
```

6.  To display the text, call `window.draw` and pass `headingText` into it. We also want the text to be drawn when the game is over. To do this, add an `if` statement, which checks whether the game is over:

```cpp
   if (gameover) { 
         window.draw(headingText); 
   }
```

7.  Run the game. You will see the name of the game displayed at the top:

![](img/510dff6d-749c-4ea1-b678-2633b15c2de8.png)

8.  We still can't see the score, so let's add a `Font` variable and a `Text` variable and call them `scoreFont` and `scoreText`, respectively. In the `scoreFont` variable, load the `arial.ttf` font and set the text for the score using the `scoreText` variable:

```cpp
sf::Font headingFont; 
sf::Text headingText; 

sf::Font scoreFont; 
sf::Text scoreText;
```

9.  Load the `ScoreFont` string and then set the `ScoreText` string:

```cpp
   scoreFont.loadFromFile("Assets/fonts/arial.ttf"); 

   // Set Score Text 

   scoreText.setFont(scoreFont); 
   scoreText.setString("Score: 0"); 
   scoreText.setCharacterSize(45); 
   scoreText.setFillColor(sf::Color::Red); 

   sf::FloatRect scorebounds = scoreText.getLocalBounds(); 
   scoreText.setOrigin(scorebounds.width / 2,
      scorebounds.height / 2); 
   scoreText.setPosition(sf::Vector2f(viewSize.x * 0.5f, 
      viewSize.y * 0.10f)); 
```

Here, we set the `scoreText` string to a score of `0`, which we will change once the score increases. Set the size of the font to `45`.

Set the score so that it's in the same position as `headingText` since it will only be displayed when the game is over. When the game is running, `scoreText` will be displayed.

10.  In the `update` function, where we update the score, update `scoreText`:

```cpp
   score++; 
   std::string finalScore = "Score: " + std::to_string(score); 
   scoreText.setString(finalScore); 
   sf::FloatRect scorebounds = scoreText.getLocalBounds(); 
   scoreText.setOrigin(scorebounds.width / 2, 
     scorebounds.height / 2); 
   scoreText.setPosition(sf::Vector2f(viewSize.x * 0.5f, viewSize.y
     * 0.10f));
```

For convenience, we created a new string called `finalScore`. Here, we set the `"Score: "` string and concatenated it with the score, which is an int that's been converted into a string by the `toString` property of the string class. Then, we used the `setString` function of `sf::Text` to set the string. We had to get the new bounds of the text since the text would have changed. Set the origin, center, and position of the updated text.

11.  In the `draw` function, create a new `else` statement. If the game is not over, draw `scoreText`:

```cpp
   if (gameover) { 
         window.draw(headingText); 
   } else { 
        window.draw(scoreText);
    } 
```

12.  Reset the `scoreText` in the `reset()` function:

```cpp
   prevTime = 0.0; 
   scoreText.setString("Score: 0"); 
```

When you run the game now, the score will continue to update. The values will reset when you restart the game.

The scoring system looks as follows:

![](img/90966be4-23ea-4a5e-a791-edfeb143c100.png)

13.  Add a tutorial so that the player knows what to do when the game starts. Create a new `sf::Text` called `tutorialText`:

```cpp
sf::Text tutorialText; 

```

14.  Initialize the text after `scoreText` in the `init()` function:

```cpp
   // Tutorial Text 

   tutorialText.setFont(scoreFont); 
   tutorialText.setString("Press Down Arrow to Fire and Start Game, 
   Up Arrow to Jump"); 
   tutorialText.setCharacterSize(35); 
   tutorialText.setFillColor(sf::Color::Red); 

   sf::FloatRect tutorialbounds = tutorialText.getLocalBounds(); 
   tutorialText.setOrigin(tutorialbounds.width / 2, tutorialbounds.height / 2); 
   tutorialText.setPosition(sf::Vector2f(viewSize.x * 0.5f, viewSize.y * 0.20f)); 
```

15.  We only want to show the tutorial at the start of the game, along with the heading text. Add the following code to the `draw` function:

```cpp
if (gameover) { 
         window.draw(headingText); 
window.draw(tutorialText);
  } 
   else { 
         window.draw(scoreText); 
   } 
```

Now, when you start the game, the player will see that the game will start if they press the down arrow key. They will also know that, when the game is running, they can press the down arrow key to shoot a rocket and use the up arrow key to jump. The following screenshot shows this on screen text:

![](img/2cde74d7-42db-418a-8c2f-633fc5ce9313.png)

# Adding audio

Let's add some audio to the game to make it a little more interesting. This will also provide audio feedback to the player to tell them whether the rocket was fired or an enemy was hit.

SFML supports `.wav` or `.ogg` files, but it doesn't support `.mp3` files. For this project, all the files will be in the `.ogg` file format as it is good for compression and is also cross-platform compatible. To start, place the audio files in the `Audio` directory in the `Assets` folder of the system. With the audio files in place, we can start playing the audio files.

Audio files can be of two types:

*   The background music, which is of a longer duration and a much higher quality than other files in the game. These files are played using the `sf::Music` class.
*   Other sound files, such as sound effects – which are smaller in size and sometimes of lower quality – are played using the `sf::Sound` class. To play the files, you also need an `sf::SoundBuffer` class, which is used to store the file and play it later.

To add audio to the game, follow these steps:

1.  Let's play the background music file, `bgMusic.ogg`. Audio files use the `Audio.hpp` header, which needs to be included at the top of the `main.cpp` file. This can be done as follows:

```cpp
 #include "SFML-2.5.1\include\SFML\Audio.hpp" 
```

2.  At the top of the `main.cpp` file, create a new instance of `sf::Music` and call it `bgMusic`:

```cpp
sf::Music bgMusic;  
```

3.  In the `init()` function, add the following lines to open the `bgMusic.ogg` file and play the `bgMusic` file:

```cpp
   // Audio  

   bgMusic.openFromFile("Assets/audio/bgMusic.ogg"); 
   bgMusic.play();
```

4.  Run the game. You will hear the background music playing as soon as the game starts.

5.  To add the sound files that are for the rockets being fired and the enemies being hit, we need two sound buffers to store both of the effects and two sound files to play the sound files. Create two variables of the `sf::SoundBuffer` type called `fireBuffer` and `hitBuffer`:

```cpp
sf::SoundBuffer fireBuffer; 
sf::SoundBuffer hitBuffer;
```

6.  Now, create two `sf::Sound` variables called `fireSound` and `hitSound`. Both can be initialized by being passed into their respective buffers, as follows:

```cpp
sf::Sound fireSound(fireBuffer); 
sf::Sound hitSound(hitBuffer); 
```

7.  In the `init` function, initialize the buffers first, as follows:

```cpp
bgMusic.openFromFile("Assets/audio/bgMusic.ogg"); 
   bgMusic.play(); 

   hitBuffer.loadFromFile("Assets/audio/hit.ogg"); 
   fireBuffer.loadFromFile("Assets/audio/fire.ogg"); 
```

8.  When the rocket intersects with the enemy, we will play the `hitSound` effect:

```cpp
hitSound.play(); 
         score++; 

         std::string finalScore = "Score: " + 
                                  std::to_string(score); 

         scoreText.setString(finalScore); 

         sf::FloatRect scorebounds = scoreText.getLocalBounds(); 
         scoreText.setOrigin(scorebounds.width / 2,
         scorebounds.height / 2); 
         scoreText.setPosition(sf::Vector2f(viewSize.x * 0.5f,
         viewSize.y * 0.10f)); 
```

9.  In the `shoot` function, we will play the `fireSound` file, as follows:

```cpp
void shoot() { 
   Rocket* rocket = new Rocket(); 

   rocket->init("Assets/graphics/rocket.png", hero.getSprite().getPosition(), 400.0f); 

   rockets.push_back(rocket); 
 fireSound.play();
} 
```

Now, when you play the game, you will hear a sound effect when you shoot the rocket and when the rocket hits the enemy.

# Adding player animations

The game has now reached its final stages of development. Let's add some animation to the game to make it really come alive. To animate 2D sprites, we need a sprite sheet. We can use other techniques to add 2D animations, such as skeletal animation, but sprite sheet-based 2D animations are faster to make. Hence, we will use sprite sheets to add animations to the main character.

A sprite sheet is an image file; however, instead of just one single image, it contains a collection of images in a sequence so that we can loop them to create the animation. Each image in the sequence is called a frame.

Here is the sprite sheet we are going to be using to animate the player:

![](img/c8e99ebf-26f3-4887-85a1-922b90f08c71.png)

Looking from left to right, we can see that each frame is slightly different from the last. The main things that are being animated here are the jet pack of the player character and the player character's eyes (so that the character looks like it's blinking). Each picture will be shown as an animation frame when the game runs, just like in a flip-book animation, where one image is quickly replaced with another image to create the effect of animation.

SFML makes it really easy to animate 2D characters since we can choose which frame to display in the `update` function. Let's start animating the character:

1.  Add the sprite sheet file to the `Assets/graphics` folder. We need to make some changes to the `Hero.h` and `Hero.cpp` files. Let's look at the changes for the `Hero.h` file first:

```cpp
class Hero{ 

public: 
   Hero(); 
   ~Hero(); 

   void init(std::string textureName, int frameCount, 
      float animDuration, sf::Vector2f position, float mass); 

void update(float dt); 
   void jump(float velocity); 
   sf::Sprite getSprite(); 

private: 

   int jumpCount = 0; 
   sf::Texture m_texture; 
   sf::Sprite m_sprite; 
   sf::Vector2f m_position; 
   float m_mass; 
   float m_velocity; 
   const float m_gravity = 9.81f; 
   bool m_grounded; 

   int m_frameCount; 
   float m_animDuration; 
   float m_elapsedTime;; 
   sf::Vector2i m_spriteSize; 

}; 
```

We need to add two more parameters to the `init` function. The first is an int called `frameCount`, which is the number of frames in the animation. In our case, there are four frames in the hero sprite sheet. The other parameter is a float, called `animDuration`, which basically sets how long you want the animation to be played. This will determine the speed of the animation.

We will also create some variables. The first two variables we'll create, `m_frameCount` and `m_animDuration`, will be used for storing `frameCount` and `animDuration` locally. We will also create a float called `m_elapsedTime`, which will keep track of how long the game has been running, and a `vector2` int called `m_spriteSize`, which will store the size of each frame.

2.  Let's move on to the `Hero.cpp` file and see what changes are needed there. Here is the modified `init` function:

```cpp
void Hero::init(std::string textureName, int frameCount, 
  float animDuration, sf::Vector2f position, float mass){ 

   m_position = position; 
   m_mass = mass; 
   m_grounded = false; 

   m_frameCount = frameCount;
   m_animDuration = animDuration;

   // Load a Texture 
   m_texture.loadFromFile(textureName.c_str()); 

   m_spriteSize = sf::Vector2i(92, 126);

   // Create Sprite and Attach a Texture 
   m_sprite.setTexture(m_texture); 
   m_sprite.setTextureRect(sf::IntRect(0, 0, m_spriteSize.x, 
     m_spriteSize.y));

   m_sprite.setPosition(m_position); 
   m_sprite.setOrigin(m_spriteSize.x / 2, m_spriteSize.y / 2);

} 
```

In the `init` function, we set `m_frameCount` and `m_animationDuration` locally. We need to hardcode the value of the width (as `92`) and height (as `126`) of each frame. If you are loading in your own images, these values will be different.

After calling `setTexture`, we will call the `setTextureRect` function of the `Sprite` class to set which part of the sprite sheet we want to display. Start at the origin of the sprite and get the first frame of the sprite sheet by passing the width and height of `spriteSHere, we passed the new heroAnim.png file instead of theize`.

Set the position and origin, which is equal to the center of the width and height of `spriteSize`.

3.  Let's make some changes to the `update` function, which is where the major magic happens:

```cpp
void Hero::update(float dt){ 
   // Animate Sprite 
   M_elapsedTime += dt; 
   int animFrame = static_cast<int> ((m_elapsedTime / 
                   m_animDuration) * m_frameCount) % m_frameCount; 

   m_sprite.setTextureRect(sf::IntRect(animFrame * m_spriteSize.x, 
      0, m_spriteSize.x, m_spriteSize.y)); 

   // Update Position 

   m_velocity -= m_mass * m_gravity * dt; 

   m_position.y -= m_velocity * dt; 

   m_sprite.setPosition(m_position); 

   if (m_position.y >= 768 * 0.75) { 

         m_position.y = 768 * 0.75; 
         m_velocity = 0; 
         m_grounded = true; 
         jumpCount = 0; 
   } 

} 
```

In the `update` function, increase the elapsed time by the delta time. Then, calculate the current animation frame number.

Update the part of the sprite sheet to be shown by calling `setTextureRect` and move the origin of the frame to the `x-axis`, which depends on `animFrame`, by multiplying it by the width of the frame. The height of the new frame doesn't change, so we set it to 0\. The width and height of the frame remain the same, so we pass in the size of the frame itself.

The rest of the functions in `Hero.cpp` remain as they are, and no changes need to be made to them.

4.  Go back to `main.cpp` so that we can change how we call `hero.init`. In the `init` function, make the required change:

```cpp
hero.init("Assets/graphics/heroAnim.png", 4, 1.0f, sf::Vector2f(viewSize.x * 0.25f, viewSize.y * 0.5f), 200); 
```

Here, we passed the new `heroAnim.png` file instead of the single-frame `.png` file we loaded previously. Set the number of frames to `4` and set `animDuration` to `1.0f`.

5.  Run the game. You will see that the player character is now animated and blinks every four frames:

![](img/f1c30d01-f217-40d9-b723-41e6206f010f.png)

# Summary

In this chapter, we completed the gameloop and added the `gameover` condition. We added scoring so that the player knows how many points they have scored. We also added text so that the name of the game is displayed, the player's score is displayed, and a tutorial is displayed that tells the user how to play the game. Then, we learned how to place these elements in the center of the viewport. Finally, we added sound effects and animations to make our game come to life.

In the next chapter, we will look at how to render 3D and 2D objects in a scene. Instead of using a framework, we will start creating a basic engine and begin our journey of understanding the basics of rendering.