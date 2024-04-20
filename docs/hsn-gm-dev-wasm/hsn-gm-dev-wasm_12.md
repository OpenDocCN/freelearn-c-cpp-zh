# 音效

网络上的音频当前处于一种混乱状态，而且已经有一段时间了。很长一段时间以来，根据您使用的浏览器的不同，加载 MP3 与 OGG 文件存在问题。最近，浏览器阻止自动播放声音以防止令人讨厌的音频垃圾的问题。Chrome 中的这一功能有时似乎会在我们的游戏中播放音频时出现问题。我注意到，如果 Chrome 最初没有播放音频，通常在重新加载页面后就会播放。我在 Firefox 上没有遇到这个问题。

您需要在构建中包含几个图像和音频文件才能使该项目正常工作。确保您从项目的 GitHub 中包含`/Chapter12/sprites/`文件夹以及`/Chapter12/audio/`文件夹。如果您还没有下载 GitHub 项目，可以在[`github.com/PacktPublishing/Hands-On-Game-Development-with-WebAssembly`](https://github.com/PacktPublishing/Hands-On-Game-Development-with-WebAssembly)上获取它。

Emscripten 对音频播放的支持并不如我所希望的那样好。在留言板上，Emscripten 的支持者很快就把音频的状态归咎于网络而不是 Emscripten 本身，这种评估有一定道理。Emscripten 的常见问题解答声称，Emscripten 支持使用 SDL1 音频、SDL2 音频和 OpenAL，但根据我的经验，我发现使用非常有限的 SDL2 音频提供了最佳的结果。我将尽量减少对 SDL2 音频的使用，使用音频队列而不是混合音效。您可能希望扩展或修改我在这里所做的工作。理论上，OpenAL 应该可以与 Emscripten 一起工作，尽管我在这方面并不太幸运。此外，您可能希望查看`SDL_MixAudio`（[`wiki.libsdl.org/SDL_MixAudio`](https://wiki.libsdl.org/SDL_MixAudio)）和`SDL_AudioStream`（[`wiki.libsdl.org/Tutorials/AudioStream`](https://wiki.libsdl.org/Tutorials/AudioStream)）来改进游戏中的音频系统，但请注意，网络上的音频流和混音的性能和支持可能还没有准备好投入实际使用。

本章将涵盖以下主题：

+   获取音效的地方

+   使用 Emscripten 制作简单音频

+   向我们的游戏添加声音

+   编译和运行

# 获取音效的地方

有很多很棒的地方可以在线获取音乐和音效。我使用 SFXR（[`www.drpetter.se/project_sfxr.html`](http://www.drpetter.se/project_sfxr.html)）生成了本章中使用的音效，这是一个用于生成类似 NES 游戏中听到的老式 8 位音效的工具。这种类型的音效可能不符合您的口味。OpenGameArt.org 还有大量的音效（[`opengameart.org/art-search-advanced?keys=&field_art_type_tid%5B%5D=13&sort_by=count&sort_order=DESC`](https://opengameart.org/art-search-advanced?keys=&field_art_type_tid%5B%5D=13&sort_by=count&sort_order=DESC)）和音乐（[`opengameart.org/art-search-advanced?keys=&field_art_type_tid%5B%5D=12&sort_by=count&sort_order=DESC`](https://opengameart.org/art-search-advanced?keys=&field_art_type_tid%5B%5D=12&sort_by=count&sort_order=DESC)）的大量开放许可，因此在使用该网站上的任何音频或艺术之前，请确保您仔细阅读许可证。

# 使用 Emscripten 制作简单音频

在我们将音效添加到主游戏之前，我将向您展示如何在`audio.c`文件中制作音频播放器，以演示**SDL 音频**如何在 WebAssembly 应用程序中用于播放音效。该应用程序将使用五种我们将在游戏中使用的音效，并允许用户按数字键 1 到 5 来播放所有选择的音效。我将首先向您展示代码分为两个部分，然后我将向您解释每一部分的功能。以下是`audio.c`中的所有代码，除了`main`函数： 

```cpp
#include <SDL2/SDL.h>
#include <emscripten.h>
#include <stdio.h>
#include <stdbool.h>

#define ENEMY_LASER "/audio/enemy-laser.wav"
#define PLAYER_LASER "/audio/player-laser.wav"
#define LARGE_EXPLOSION "/audio/large-explosion.wav"
#define SMALL_EXPLOSION "/audio/small-explosion.wav"
#define HIT "/audio/hit.wav"

SDL_AudioDeviceID device_id;
SDL_Window *window;
SDL_Renderer *renderer;
SDL_Event event;

struct audio_clip {
    char file_name[100];
    SDL_AudioSpec spec;
    Uint32 len;
    Uint8 *buf;
} enemy_laser_snd, player_laser_snd, small_explosion_snd, large_explosion_snd, hit_snd;

void play_audio( struct audio_clip* clip ) {
    int success = SDL_QueueAudio(device_id, clip->buf, clip->len);
    if( success < 0 ) {
        printf("SDL_QueueAudio %s failed: %s\n", clip->file_name, 
        SDL_GetError());
    }
}

void init_audio( char* file_name, struct audio_clip* clip ) {
    strcpy( clip->file_name, file_name );

    if( SDL_LoadWAV(file_name, &(clip->spec), &(clip->buf), &(clip->len)) 
    == NULL ) {
        printf("Failed to load wave file: %s\n", SDL_GetError());
    }
}

void input_loop() {
    if( SDL_PollEvent( &event ) ){
        if( event.type == SDL_KEYUP ) {
            switch( event.key.keysym.sym ){
                case SDLK_1:
                    printf("one key release\n");
                    play_audio(&enemy_laser_snd);
                    break;
                case SDLK_2:
                    printf("two key release\n");
                    play_audio(&player_laser_snd);
                    break;
                case SDLK_3:
                    printf("three key release\n");
                    play_audio(&small_explosion_snd);
                    break;
                case SDLK_4:
                    printf("four key release\n");
                    play_audio(&large_explosion_snd);
                    break;
                case SDLK_5:
                    printf("five key release\n");
                    play_audio(&hit_snd);
                    break;
                default:
                    printf("unknown key release\n");
                    break;
            }
        }
    }
}
```

在`audio.c`文件的末尾，我们有我们的`main`函数：

```cpp
int main() {
    if((SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO)==-1)) {
        printf("Could not initialize SDL: %s.\n", SDL_GetError());
        return 0;
    }

    SDL_CreateWindowAndRenderer( 320, 200, 0, &window, &renderer );

    init_audio( ENEMY_LASER, &enemy_laser_snd );
    init_audio( PLAYER_LASER, &player_laser_snd );
    init_audio( SMALL_EXPLOSION, &small_explosion_snd );
    init_audio( LARGE_EXPLOSION, &large_explosion_snd );
    init_audio( HIT, &hit_snd );

    device_id = SDL_OpenAudioDevice(NULL, 0, &(enemy_laser_snd.spec), 
                                    NULL, 0);

    if (device_id == 0) {
        printf("Failed to open audio: %s\n", SDL_GetError());
    }

    SDL_PauseAudioDevice(device_id, 0);

    emscripten_set_main_loop(input_loop, 0, 0);

    return 1;
}
```

现在你已经看到了整个`audio.c`文件，让我们来看看它的所有部分。在这个文件的顶部，我们有我们的`#include`和`#define`宏：

```cpp
#include <SDL2/SDL.h>
#include <emscripten.h>
#include <stdio.h>
#include <stdbool.h>

#define ENEMY_LASER "/audio/enemy-laser.wav"
#define PLAYER_LASER "/audio/player-laser.wav"
#define LARGE_EXPLOSION "/audio/large-explosion.wav"
#define SMALL_EXPLOSION "/audio/small-explosion.wav"
#define HIT "/audio/hit.wav"
```

之后，我们有我们的 SDL 特定的全局变量。我们需要一个`SDL_AudioDeviceID`用于我们的音频输出。`SDL_Window`、`SDL_Renderer`和`SDL_Event`在大多数早期章节中都被使用过，现在应该很熟悉了：

```cpp
SDL_AudioDeviceID device_id;
SDL_Window *window;
SDL_Renderer *renderer;
SDL_Event event;
```

我们正在开发一个 C 程序，而不是 C++程序，所以我们将使用一个结构来保存我们的音频数据，而不是一个类。我们将创建一个名为`audio_clip`的 C 结构，它将保存我们应用程序中将要播放的音频的所有信息。这些信息包括一个包含文件名的字符串。它包含一个保存音频规格的`SDL_AudioSpec`对象。它还包含音频片段的长度和一个指向 8 位数据缓冲区的指针，该缓冲区保存了音频片段的波形数据。在定义了`audio_clip`结构之后，创建了五个该结构的实例，我们稍后将能够使用这些声音进行播放：

```cpp
struct audio_clip {
    char file_name[100];
    SDL_AudioSpec spec;
    Uint32 len;
    Uint8 *buf;
} enemy_laser_snd, player_laser_snd, small_explosion_snd, large_explosion_snd, hit_snd;
```

在我们定义了`audio_clip`结构之后，我们需要创建一个函数来播放该结构中的音频。这个函数调用`SDL_QueueAudio`，传入全局`device_id`、波形缓冲区的指针和片段的长度。`device_id`是对音频设备（声卡）的引用。`clip->buf`变量是一个指向包含我们将要加载的`.wav`文件的波形数据的缓冲区的指针。`clip->len`变量包含片段播放的时间长度：

```cpp
void play_audio( struct audio_clip* clip ) {
    int success = SDL_QueueAudio(device_id, clip->buf, clip->len);
    if( success < 0 ) {
        printf("SDL_QueueAudio %s failed: %s\n", clip->file_name, 
        SDL_GetError());
    }
}
```

我们需要的下一个函数是初始化我们的`audio_clip`，这样我们就可以将它传递到`play_audio`函数中。这个函数设置了我们的`audio_clip`的文件名，并加载了一个波形文件，设置了我们的`audio_clip`中的`spec`、`buf`和`len`值。如果调用`SDL_LoadWAV`失败，我们会打印出一个错误消息：

```cpp
void init_audio( char* file_name, struct audio_clip* clip ) {
    strcpy( clip->file_name, file_name );

    if( SDL_LoadWAV(file_name, &(clip->spec), &(clip->buf), &(clip-
        >len)) 
    == NULL ) {
        printf("Failed to load wave file: %s\n", SDL_GetError());
    }
}
```

`input_loop`现在应该看起来很熟悉了。该函数调用`SDL_PollEvent`并使用它返回的事件来检查键盘按键的释放。它检查释放了哪个键。如果该键是从一到五的数字键之一，那么使用 switch 语句调用`play_audio`函数，传入特定的`audio_clip`。我们使用按键释放而不是按键按下的原因是为了防止用户按住键时的按键重复。我们可以很容易地防止这种情况，但我正在尽量保持这个应用程序的代码尽可能简短。这是`input_loop`的代码：

```cpp
void input_loop() {
    if( SDL_PollEvent( &event ) ){
        if( event.type == SDL_KEYUP ) {
            switch( event.key.keysym.sym ){
                case SDLK_1:
                    printf("one key release\n");
                    play_audio(&enemy_laser_snd);
                    break;
                case SDLK_2:
                    printf("two key release\n");
                    play_audio(&player_laser_snd);
                    break;
                case SDLK_3:
                    printf("three key release\n");
                    play_audio(&small_explosion_snd);
                    break;
                case SDLK_4:
                    printf("four key release\n");
                    play_audio(&large_explosion_snd);
                    break;
                case SDLK_5:
                    printf("five key release\n");
                    play_audio(&hit_snd);
                    break;
                default:
                    printf("unknown key release\n");
                    break;
            }
        }
    }
}
```

和往常一样，`main`函数负责我们应用程序的所有初始化。除了我们在之前的应用程序中执行的初始化之外，我们还需要对我们的音频进行新的初始化。这就是`main`函数的新版本。

```cpp
int main() {
    if((SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO)==-1)) {
        printf("Could not initialize SDL: %s.\n", SDL_GetError());
        return 0;
    }
    SDL_CreateWindowAndRenderer( 320, 200, 0, &window, &renderer );
    init_audio( ENEMY_LASER, &enemy_laser_snd );
    init_audio( PLAYER_LASER, &player_laser_snd );
    init_audio( SMALL_EXPLOSION, &small_explosion_snd );
    init_audio( LARGE_EXPLOSION, &large_explosion_snd );
    init_audio( HIT, &hit_snd );

    device_id = SDL_OpenAudioDevice(NULL, 0, &(enemy_laser_snd.spec), NULL, 
    0);

    if (device_id == 0) {
        printf("Failed to open audio: %s\n", SDL_GetError());
    }
    SDL_PauseAudioDevice(device_id, 0);
    emscripten_set_main_loop(input_loop, 0, 0);
    return 1;
}
```

我们改变的第一件事是我们对`SDL_Init`的调用。我们需要添加一个标志，告诉 SDL 初始化音频子系统。我们通过在传入的参数中添加`|SLD_INIT_AUDIO`来实现这一点，这将对参数进行位操作，并使用`SDL_INIT_AUDIO`标志。在新版本的`SDL_Init`之后，我们将创建窗口和渲染器，这在这一点上我们已经做了很多次。

`init_audio`调用都是新的，并初始化了我们的`audio_clip`结构：

```cpp
init_audio( ENEMY_LASER, &enemy_laser_snd );
init_audio( PLAYER_LASER, &player_laser_snd );
init_audio( SMALL_EXPLOSION, &small_explosion_snd );
init_audio( LARGE_EXPLOSION, &large_explosion_snd );
init_audio( HIT, &hit_snd );
```

接下来，我们需要调用`SDL_OpenAudioDevice`并检索设备 ID。打开音频设备需要一个默认规范，它通知音频设备您想要播放的声音剪辑的质量。确保选择一个声音文件，其质量水平是您想在游戏中播放的一个很好的例子。在我们的代码中，我们选择了`enemy_laser_snd`。我们还需要调用`SDL_PauseAudioDevice`。每当创建新的音频设备时，默认情况下会暂停。调用`SDL_PauseAudioDevice`并将`0`作为第二个参数传递进去会取消暂停我们刚刚创建的音频设备。起初我觉得有点困惑，但请记住，对`SDL_PauseAudioDevice`的后续调用实际上是取消暂停音频剪辑：

```cpp
device_id = SDL_OpenAudioDevice(NULL, 0, &(enemy_laser_snd.spec), NULL, 0);

if (device_id == 0) {
    printf("Failed to open audio: %s\n", SDL_GetError());
}

SDL_PauseAudioDevice(device_id, 0);
```

在返回之前，我们将做的最后一件事是将我们的循环设置为我们之前创建的`input_loop`函数：

```cpp
emscripten_set_main_loop(input_loop, 0, 0);
```

现在我们有了代码，我们应该编译和测试我们的`audio.c`文件：

```cpp
emcc audio.c --preload-file audio -s USE_SDL=2 -o audio.html
```

我们需要预加载音频文件夹，以便在虚拟文件系统中访问`.wav`文件。然后，在 Web 浏览器中加载`audio.html`，使用 emrun 提供文件，或者使用其他替代 Web 服务器。当您在 Chrome 中加载应用程序时，可能会遇到一些小困难。Chrome 的新版本已添加了检查，以防止未经请求的音频播放，以防止一些令人讨厌的垃圾邮件。有时，这种检查过于敏感，这可能会阻止我们游戏中的音频运行。如果发生这种情况，请尝试在 Chrome 浏览器中重新加载页面。有时，这可以解决问题。另一种防止这种情况发生的方法是切换到 Firefox。

# 向我们的游戏添加声音

现在我们了解了如何在 Web 上让 SDL 音频工作，我们可以开始向我们的游戏添加音效。我们的游戏中不会使用混音器，因此一次只会播放一个音效。因此，我们需要将一些声音分类为**优先**音效。如果触发了优先音效，声音队列将被清除，并且该音效将运行。我们还希望防止我们的声音队列变得太长，因此如果其中有两个以上的项目，我们将清除我们的声音队列。不要害怕！当我们到达代码的那部分时，我会重复所有这些。

# 更新 game.hpp

我们需要改变的第一件事是我们的`game.hpp`文件。我们需要添加一个新的`Audio`类，以及其他新代码来支持我们游戏中的音频。在`game.hpp`文件的顶部附近，我们将添加一系列`#define`宏来定义我们声音效果`.wav`文件的位置：

```cpp
#define ENEMY_LASER (char*)"/audio/enemy-laser.wav"
#define PLAYER_LASER (char*)"/audio/player-laser.wav"
#define LARGE_EXPLOSION (char*)"/audio/large-explosion.wav"
#define SMALL_EXPLOSION (char*)"/audio/small-explosion.wav"
#define HIT (char*)"/audio/hit.wav"
```

在我们的类声明列表的顶部，我们应该添加一个名为`Audio`的新类声明：

```cpp
class Audio;
class Ship;
class Particle;
class Emitter;
class Collider;
class Asteroid;
class Star;
class PlayerShip;
class EnemyShip;
class Projectile;
class ProjectilePool;
class FiniteStateMachine;
class Camera;
class RenderManager;
class Locator;
```

然后，我们将定义新的`Audio`类，它将与我们在`audio.c`文件中使用的`audio_clip`结构非常相似。这个类将有一个文件名，一个规范，一个长度（以运行时间为单位）和一个缓冲区。它还将有一个优先标志，当设置时，将优先于我们音频队列中当前的所有其他内容。最后，我们将在这个类中有两个函数；一个构造函数，用于初始化声音，和一个`Play`函数，用于实际播放声音。这就是类定义的样子：

```cpp
class Audio {
    public:
        char FileName[100];
        SDL_AudioSpec spec;
        Uint32 len;
        Uint8 *buf;
        bool priority = false;

        Audio( char* file_name, bool priority_value );
        void Play();
};
```

最后，我们需要定义一些外部与音频相关的全局变量。这些全局变量将是对将出现在我们的`main.cpp`文件中的变量的引用。其中大部分是`Audio`类的实例，将在我们的游戏中用于播放音频文件。最后一个变量是对我们的音频设备的引用：

```cpp
extern Audio* enemy_laser_snd;
extern Audio* player_laser_snd;
extern Audio* small_explosion_snd;
extern Audio* large_explosion_snd;
extern Audio* hit_snd;
extern SDL_AudioDeviceID device_id;
```

# 更新 main.cpp

在我们的`main.cpp`文件中要做的第一件事是定义我们在`game.hpp`文件的末尾定义为外部变量的与音频相关的全局变量：

```cpp
SDL_AudioDeviceID device_id;

Audio* enemy_laser_snd;
Audio* player_laser_snd;
Audio* small_explosion_snd;
Audio* large_explosion_snd;
Audio* hit_snd;
```

这些音效大多与我们游戏中发生碰撞时爆炸有关。因此，我们将在整个`collisions`函数中添加调用以播放这些音效。这是我们`collisions`函数的新版本：

```cpp
void collisions() {
 Asteroid* asteroid;
 std::vector<Asteroid*>::iterator ita;
    if( player->m_CurrentFrame == 0 && player->CompoundHitTest( star ) ) {
        player->m_CurrentFrame = 1;
        player->m_NextFrameTime = ms_per_frame;
        player->m_Explode->Run(); // added
        large_explosion_snd->Play();
    }
    if( enemy->m_CurrentFrame == 0 && enemy->CompoundHitTest( star ) ) {
        enemy->m_CurrentFrame = 1;
        enemy->m_NextFrameTime = ms_per_frame;
        enemy->m_Explode->Run(); // added
        large_explosion_snd->Play();
    }
    Projectile* projectile;
    std::vector<Projectile*>::iterator it;
    for(it=projectile_pool->m_ProjectileList.begin(); 
        it!=projectile_pool->m_ProjectileList.end(); 
        it++){
        projectile = *it;
        if( projectile->m_CurrentFrame == 0 && projectile->m_Active ) {
            for( ita = asteroid_list.begin(); ita != 
                asteroid_list.end(); 
                 ita++ ) {
                asteroid = *ita;
                if( asteroid->m_Active ) {
                    if( asteroid->HitTest( projectile ) ) {
                        projectile->m_CurrentFrame = 1;
                        projectile->m_NextFrameTime = ms_per_frame;
                        small_explosion_snd->Play();
                    }
                }
            }
            if( projectile->HitTest( star ) ){
                projectile->m_CurrentFrame = 1;
                projectile->m_NextFrameTime = ms_per_frame;
                small_explosion_snd->Play();
            }
            else if( player->m_CurrentFrame == 0 && ( projectile-
                     >HitTest( player ) ||
                      player->CompoundHitTest( projectile ) ) ) {
                if( player->m_Shield->m_Active == false ) {
                    player->m_CurrentFrame = 1;
                    player->m_NextFrameTime = ms_per_frame;
                    player->m_Explode->Run();
                    large_explosion_snd->Play();
                }
                else { hit_snd->Play(); }
                projectile->m_CurrentFrame = 1;
                projectile->m_NextFrameTime = ms_per_frame;
            }
            else if( enemy->m_CurrentFrame == 0 && ( projectile-
                     >HitTest( enemy ) ||
                      enemy->CompoundHitTest( projectile ) ) ) {
                if( enemy->m_Shield->m_Active == false ) {
                    enemy->m_CurrentFrame = 1;
                    enemy->m_NextFrameTime = ms_per_frame;
                    enemy->m_Explode->Run();
                    large_explosion_snd->Play();
                }
                else { hit_snd->Play(); }
                projectile->m_CurrentFrame = 1;
                projectile->m_NextFrameTime = ms_per_frame;
            }
        }
    }
    for( ita = asteroid_list.begin(); ita != asteroid_list.end(); 
         ita++ ) {
        asteroid = *ita;
        if( asteroid->m_Active ) {
            if( asteroid->HitTest( star ) ) {
                asteroid->Explode();
                small_explosion_snd->Play();
            }
        }
        else { continue; }
        if( player->m_CurrentFrame == 0 && asteroid->m_Active &&
            ( asteroid->HitTest( player ) || player->CompoundHitTest( 
            asteroid ) ) ) {
            if( player->m_Shield->m_Active == false ) {
                player->m_CurrentFrame = 1;
                player->m_NextFrameTime = ms_per_frame;
                player->m_Explode->Run();
                large_explosion_snd->Play();
            }
            else {
                asteroid->Explode();
                small_explosion_snd->Play();
            }
        }
        if( enemy->m_CurrentFrame == 0 && asteroid->m_Active &&
            ( asteroid->HitTest( enemy ) || enemy->CompoundHitTest( 
              asteroid ) ) ) {
            if( enemy->m_Shield->m_Active == false ) {
                enemy->m_CurrentFrame = 1;
                enemy->m_NextFrameTime = ms_per_frame;
                enemy->m_Explode->Run();
                large_explosion_snd->Play();
            }
            else {
                asteroid->Explode();
                small_explosion_snd->Play();
            }
        }
    }
}
```

现在声音将在几次爆炸和碰撞后播放；例如，在玩家爆炸后：

```cpp
player->m_Explode->Run(); 
large_explosion_snd->Play();
```

当敌舰爆炸时也会播放声音：

```cpp
enemy->m_Explode->Run();
large_explosion_snd->Play();
```

在一颗小行星爆炸后，我们也希望有同样的效果：

```cpp
asteroid->Explode();
small_explosion_snd->Play();
```

如果敌人的护盾被击中，我们想播放`hit`声音：

```cpp
if( enemy->m_Shield->m_Active == false ) {
    enemy->m_CurrentFrame = 1;
    enemy->m_NextFrameTime = ms_per_frame;
    enemy->m_Explode->Run();
    large_explosion_snd->Play();
}
else {
    hit_snd->Play();
}
```

同样，如果玩家的护盾被击中，我们还想播放`hit`声音：

```cpp
if( player->m_Shield->m_Active == false ) {
    player->m_CurrentFrame = 1;
    player->m_NextFrameTime = ms_per_frame;

    player->m_Explode->Run();
    large_explosion_snd->Play();
}
else {
    hit_snd->Play();
}
```

最后，我们需要更改`main`函数来初始化我们的音频。以下是完整的`main`函数代码：

```cpp
int main() {
    SDL_Init( SDL_INIT_VIDEO | SDL_INIT_AUDIO );
    int return_val = SDL_CreateWindowAndRenderer( CANVAS_WIDTH, 
    CANVAS_HEIGHT, 0, &window, &renderer );

    if( return_val != 0 ) {
        printf("Error creating renderer %d: %s\n", return_val, 
        IMG_GetError() );
        return 0;
    }

    SDL_SetRenderDrawColor( renderer, 0, 0, 0, 255 );
    SDL_RenderClear( renderer );
    last_frame_time = last_time = SDL_GetTicks();

    player = new PlayerShip();
    enemy = new EnemyShip();
    star = new Star();
    camera = new Camera(CANVAS_WIDTH, CANVAS_HEIGHT);
    render_manager = new RenderManager();
    locator = new Locator();
    enemy_laser_snd = new Audio(ENEMY_LASER, false);
 player_laser_snd = new Audio(PLAYER_LASER, false);
 small_explosion_snd = new Audio(SMALL_EXPLOSION, true);
 large_explosion_snd = new Audio(LARGE_EXPLOSION, true);
 hit_snd = new Audio(HIT, false);
 device_id = SDL_OpenAudioDevice(NULL, 0, &(enemy_laser_snd->spec), 
    NULL, 0);

 if (device_id == 0) {
 printf("Failed to open audio: %s\n", SDL_GetError());
 }
    int asteroid_x = 0;
    int asteroid_y = 0;
    int angle = 0;

    // SCREEN 1
    for( int i_y = 0; i_y < 8; i_y++ ) {
        asteroid_y += 100;
        asteroid_y += rand() % 400;
        asteroid_x = 0;
        for( int i_x = 0; i_x < 12; i_x++ ) {
            asteroid_x += 66;
            asteroid_x += rand() % 400;
            int y_save = asteroid_y;
            asteroid_y += rand() % 400 - 200;
            angle = rand() % 359;
            asteroid_list.push_back(
                new Asteroid( asteroid_x, asteroid_y,
                get_random_float(0.5, 1.0),
                DEG_TO_RAD(angle) ) );
            asteroid_y = y_save;
        }
    }
    projectile_pool = new ProjectilePool();
    emscripten_set_main_loop(game_loop, 0, 0);
    return 1;
}
```

我们需要对`main`函数进行的第一个更改是在`SDL_Init`调用中包括音频子系统的初始化：

```cpp
SDL_Init( SDL_INIT_VIDEO | SDL_INIT_AUDIO );
```

我们需要做的另一个更改是添加新的`Audio`对象和调用`SDL_OpenAudioDevice`：

```cpp
enemy_laser_snd = new Audio(ENEMY_LASER, false);
player_laser_snd = new Audio(PLAYER_LASER, false);
small_explosion_snd = new Audio(SMALL_EXPLOSION, true);
large_explosion_snd = new Audio(LARGE_EXPLOSION, true);
hit_snd = new Audio(HIT, false);

device_id = SDL_OpenAudioDevice(NULL, 0, &(enemy_laser_snd->spec), 
NULL, 0);

if (device_id == 0) {
    printf("Failed to open audio: %s\n", SDL_GetError());
}
```

# 更新 ship.cpp

`ship.cpp`文件有一个小的更改。我们正在添加一个调用，当飞船发射抛射物时播放声音。这发生在`Ship::Shoot()`函数中。您会注意到在调用`projectile->Launch`之后发生对`player_laser_snd->Play()`的调用：

```cpp
void Ship::Shoot() {
     Projectile* projectile;
     if( current_time - m_LastLaunchTime >= c_MinLaunchTime ) {
         m_LastLaunchTime = current_time;
         projectile = projectile_pool->GetFreeProjectile();
         if( projectile != NULL ) {
             projectile->Launch( m_Position, m_Direction );
             player_laser_snd->Play();
         }
     }
 }
```

# 新的 audio.cpp 文件

我们正在添加一个新的`audio.cpp`文件来实现`Audio`类的构造函数和`Audio`类的`Play`函数。以下是完整的`audio.cpp`文件：

```cpp
#include "game.hpp"

Audio::Audio( char* file_name, bool priority_value ) {
    strcpy( FileName, file_name );
    priority = priority_value;

    if( SDL_LoadWAV(FileName, &spec, &buf, &len) == NULL ) {
        printf("Failed to load wave file: %s\n", SDL_GetError());
    }
}

void Audio::Play() {
    if( priority || SDL_GetQueuedAudioSize(device_id) > 2 ) {
        SDL_ClearQueuedAudio(device_id);
    }

    int success = SDL_QueueAudio(device_id, buf, len);
    if( success < 0 ) {
        printf("SDL_QueueAudio %s failed: %s\n", FileName, SDL_GetError());
    }
}
```

该文件中的第一个函数是`Audio`类的构造函数。此函数将`FileName`属性设置为传递的值，并设置`priority`值。它还从传递的文件名加载波形文件，并使用`SDL_LoadWAV`文件设置`spec`、`buf`和`len`属性。

`Audio::Play()`函数首先查看这是否是高优先级音频，或者音频队列的大小是否大于两个声音。如果是这种情况，我们会清空音频队列：

```cpp
if( priority || SDL_GetQueuedAudioSize(device_id) > 2 ) {
    SDL_ClearQueuedAudio(device_id);
}
```

我们这样做是因为我们不想混合音频。我们正在按顺序播放音频。如果我们有一个优先级音频剪辑，我们希望清空队列，以便音频立即播放。如果队列太长，我们也希望这样做。然后我们将调用`SDL_QueueAudio`来排队播放此声音以尽快播放：

```cpp
int success = SDL_QueueAudio(device_id, buf, len);
if( success < 0 ) {
 printf("SDL_QueueAudio %s failed: %s\n", FileName, SDL_GetError());
}
```

现在，我们应该准备编译和运行我们的代码。

# 编译和运行

现在我们已经对我们的代码进行了所有必要的更改，我们可以使用 Emscripten 编译和运行我们的新代码：

```cpp
em++ asteroid.cpp audio.cpp camera.cpp collider.cpp emitter.cpp enemy_ship.cpp finite_state_machine.cpp locator.cpp main.cpp particle.cpp player_ship.cpp projectile_pool.cpp projectile.cpp range.cpp render_manager.cpp shield.cpp ship.cpp star.cpp vector.cpp -o sound_fx.html --preload-file audio --preload-file sprites -std=c++17 -s USE_WEBGL2=1 -s USE_SDL=2 -s USE_SDL_IMAGE=2 -s SDL2_IMAGE_FORMATS=["png"] -s USE_SDL_IMAGE=2 -s SDL2_IMAGE_FORMATS=["png"] 
```

没有添加新的标志来允许我们使用 SDL 音频库。但是，我们需要添加一个新的`--preload-file audio`标志，将新的`audio`目录加载到我们的虚拟文件系统中。一旦编译了游戏的新版本，您可以使用 emrun 来运行它（假设您在编译时包含了必要的 emrun 标志）。如果您愿意，您也可以选择一个不同的 Web 服务器来提供这些文件。

# 总结

我们已经讨论了网络上当前（混乱的）音频状态，并查看了 Emscripten 可用的音频库。我提到了一些可以获得免费音效的地方。我们使用 C 和 Emscripten 创建了一个简单的音频应用程序，允许我们播放一系列音频文件。然后我们为我们的游戏添加了音效，包括爆炸和激光声音。我们修改了`main()`函数中的初始化代码，以初始化 SDL 音频子系统。我们添加了一个新的`Shoot`函数，供我们的飞船在发射抛射物时使用。我们还创建了一个新的`Audio`类来帮助我们播放我们的音频文件。

在下一章中，我们将学习如何为我们的游戏添加一些物理效果。
