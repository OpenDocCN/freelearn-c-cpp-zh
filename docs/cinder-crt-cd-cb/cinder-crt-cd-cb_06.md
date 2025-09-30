# 第六章. 渲染和纹理化粒子系统

在本章中，我们将学习以下内容：

+   纹理化粒子

+   为我们的粒子添加尾巴

+   创建布料模拟

+   纹理化布料模拟

+   使用点精灵和着色器纹理化粒子系统

+   连接粒子

# 简介

从第五章，*构建粒子系统*继续，我们将学习如何渲染和将纹理应用到我们的粒子上，以使它们更具吸引力。

# 纹理化粒子

在本食谱中，我们将使用从 PNG 文件加载的纹理来渲染上一章中引入的粒子。

## 开始

本食谱的代码库是第五章，*构建粒子系统*中*模拟粒子随风飘动*的食谱示例。我们还需要一个单个粒子的纹理。你可以用任何图形程序轻松制作一个。对于这个例子，我们将使用存储在`assets`文件夹中名为`particle.png`的 PNG 文件。在这种情况下，它只是一个带有透明度的径向渐变。

![开始](img/8703OS_06_01.jpg)

## 如何做…

我们将使用之前创建的纹理来渲染粒子。

1.  包含必要的头文件：

    ```cpp
    #include "cinder/gl/Texture.h"
    #include "cinder/ImageIo.h"
    ```

1.  向应用程序主类添加一个成员：

    ```cpp
    gl::Texture particleTexture;
    ```

1.  在`setup`方法中加载`particleTexture`：

    ```cpp
    particleTexture=gl::Texture(loadImage(loadAsset("particle.png")));
    ```

1.  我们还必须更改此示例中的粒子大小：

    ```cpp
    float radius = Rand::randFloat(2.f, 10.f);
    ```

1.  在`draw`方法结束时，我们将按照以下方式绘制我们的粒子：

    ```cpp
    gl::enableAlphaBlending();
    particleTexture.enableAndBind();
    gl::color(ColorA::white());
    mParticleSystem.draw();
    ```

1.  将`Particle.cpp`源文件内的`draw`方法替换为以下代码：

    ```cpp
    void Particle::draw(){
    ci::gl::drawSolidRect(ci::Rectf(position.x-radius, position.y-radius,
    position.x+radius, position.y+radius));
    }
    ```

## 它是如何工作的…

在步骤 5 中，我们看到了两条重要的行。一条启用了 alpha 混合，另一条将存储在`particleTexture`属性中的纹理绑定。如果你看步骤 6，你可以看到我们以矩形的形式绘制了每个粒子，每个矩形都应用了纹理。这是一种简单的纹理化粒子的方法，但不是非常高效，但在这个例子中，它相当有效。通过在调用`ParticleSystem`上的`draw`方法之前更改颜色，可以更改绘制粒子的颜色。

![如何工作…](img/8703OS_06_02.jpg)

## 参见

查看食谱*使用点精灵和着色器纹理化粒子系统*

# 为我们的粒子添加尾巴

在本食谱中，我们将向您展示如何为粒子动画添加尾巴。

## 开始

在本食谱中，我们将使用来自第五章，*构建粒子系统*中*应用排斥和吸引力的力*的代码库。

## 如何做…

我们将使用不同的技术为粒子添加尾巴。

### 绘制历史

简单地替换`draw`方法为以下代码：

```cpp
void MainApp::draw()
{   
gl::enableAlphaBlending();
gl::setViewport(getWindowBounds());
gl::setMatricesWindow(getWindowWidth(), getWindowHeight());

gl::color( ColorA(0.f,0.f,0.f, 0.05f) );
gl::drawSolidRect(getWindowBounds());
gl::color( ColorA(1.f,1.f,1.f, 1.f) );
mParticleSystem.draw();
}
```

### 尾巴作为线条

我们将添加由几条线构成的尾巴。

1.  在`Particle.h`头文件内的`Particle`类中添加新的属性：

    ```cpp
    std::vector<ci::Vec2f> positionHistory;
    int tailLength;
    ```

1.  在`Particle`构造函数的末尾，在`Particle.cpp`源文件中，将`tailLength`属性的默认值设置为：

    ```cpp
    tailLength = 10;
    ```

1.  在`Particle`类的`update`方法末尾添加以下代码：

    ```cpp
    position History.push_back(position);
    if(positionHistory.size() >tailLength) {
    positionHistory.erase( positionHistory.begin() );
    }
    ```

1.  将您的`Particle::draw`方法替换为以下代码：

    ```cpp
    void Particle::draw(){
      glBegin( GL_LINE_STRIP );
      for( int i=0; i<positionHistory.size(); i++ ){
    float alpha = (float)i/(float)positionHistory.size();
    ci::gl::color( ci::ColorA(1.f,1.f,1.f, alpha));
    ci::gl::vertex( positionHistory[i] );
      }
      glEnd();

    ci::gl::color( ci::ColorA(1.f,1.f,1.f, 1.f) );
    ci::gl::drawSolidCircle( position, radius );
    }
    ```

# 工作原理…

现在，我们将解释每种技术是如何工作的。

## 绘制历史

这种方法的背后思想非常简单，我们不是清除绘图区域，而是连续绘制半透明的矩形，这些矩形越来越多地覆盖旧的绘图状态。这种方法非常简单，但可以给粒子带来有趣的效果。您还可以通过更改矩形的 alpha 通道来操纵每个矩形的透明度，这将成为背景的颜色。

![绘制历史](img/8703OS_06_03.jpg)

## 尾巴作为线条

要用线条绘制尾巴，我们必须存储几个粒子位置，并通过这些位置绘制具有可变透明度的线条。透明度的规则只是用较低的透明度绘制较旧的位置。您可以在步骤 4 中看到绘图代码和 alpha 通道计算。

![尾巴作为线条](img/8703OS_06_04.jpg)

# 创建布料模拟

在这个食谱中，我们将学习如何通过创建由弹簧连接的粒子网格来模拟布料。

## 准备工作

在这个食谱中，我们将使用第五章中描述的粒子系统，*构建粒子系统*中的“在 2D 中创建粒子系统”。

我们还将使用在第五章中通过“创建弹簧”食谱创建的`Springs`类，*构建粒子系统*。

因此，您需要将以下文件添加到您的项目中：

+   `Particle.h`

+   `ParticleSystem.h`

+   `Spring.h`

+   `Spring.cpp`

## 如何操作…

我们将创建一个由弹簧连接的粒子网格来创建布料模拟。

1.  通过在源文件顶部添加以下代码将粒子系统文件包含到您的项目中：

    ```cpp
    #include "ParticleSystem.h"
    ```

1.  在应用程序类声明之前添加`using`语句，如下所示：

    ```cpp
    using namespace ci;
    using namespace ci::app;
    using namespace std;
    ```

1.  创建一个`ParticleSystem`对象实例和存储网格顶角的成员变量。我们还将创建存储组成我们网格的行数和列数的变量。在您的应用程序类中添加以下代码：

    ```cpp
    ParticleSystem mParticleSystem;
      Vec2f mLeftCorner, 
    mRightCorner;
      intmNumRows, mNumLines;
    ```

1.  在我们开始创建我们的粒子网格之前，让我们更新并绘制我们的应用程序中的`update`和`draw`方法中的粒子系统。

    ```cpp
    Void MyApp::update(){
      mParticleSystem.update();
    }

    void MyApp::draw(){
      gl::clear( Color( 0, 0, 0 ) ); 
      mParticleSystem.draw();
    }
    ```

1.  在`setup`方法中，让我们初始化网格角落位置和行数。在`setup`方法的顶部添加以下代码：

    ```cpp
    mLeftCorner = Vec2f( 50.0f, 50.0f );
    mRightCorner = Vec2f( getWindowWidth() - 50.0f, 50.0f );
    mNumRows = 20;
    mNumLines = 15;
    ```

1.  计算网格上每个粒子之间的距离。

    ```cpp
    float gap = ( mRightCorner.x - mLeftCorner.x ) / ( mNumRows-1 );
    ```

1.  让我们创建一个均匀分布的粒子网格并将它们添加到`ParticleSystem`中。我们将通过创建一个嵌套循环来实现这一点，其中每个循环索引将用于计算粒子的位置。在`setup`方法中添加以下代码：

    ```cpp
    for( int i=0; i<mNumRows; i++ ){
    for( int j=0; j<mNumLines; j++ ){
    float x = mLeftCorner.x + ( gap * i );
    float y = mLeftCorner.y + ( gap * j );
    Particle *particle = new Particle( Vec2f( x, y ), 5.0f, 5.0f, 0.95f );
    mParticleSystem.addParticle( particle );
            }
        }
    ```

1.  现在粒子已经创建，我们需要用弹簧将它们连接起来。让我们首先将每个粒子与其正下方的粒子连接起来。在一个嵌套循环中，我们将计算`ParticleSystem`中粒子的索引以及它下面的粒子的索引。然后我们创建一个`Spring`类，使用它们的当前距离作为`rest`和`strength`值为`1.0`来连接这两个粒子。将以下内容添加到`setup`方法的底部：

    ```cpp
    for( int i=0; i<mNumRows; i++ ){
    for( int j=0; j<mNumLines-1; j++ ){
    int indexA = i * mNumLines + j;
    int indexB = i * mNumLines + j + 1;
                Particle *partA = mParticleSystem.particles[ indexA ];
                Particle *partB = mParticleSystem.particles[ indexB ];
    float rest = partA->position.distance( partB->position );
                Spring *spring = new Spring( partA, partB, rest, 1.0f );
    mParticleSystem.addSpring( spring );
            }
        }
    ```

1.  现在我们有一个由粒子和弹簧组成的静态网格。让我们通过向每个粒子应用一个恒定的垂直力来添加一些重力。将以下代码添加到`update`方法的底部：

    ```cpp
    Vec2f gravity( 0.0f, 1.0f );
    for( vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ){
            (*it)->forces += gravity;
        }
    ```

1.  为了防止网格向下坠落，我们需要将顶边界的粒子在它们的初始位置（由`mLeftCorner`和`mRightCorner`定义）设置为静态。将以下代码添加到`update`方法中：

    ```cpp
    int topLeftIndex = 0;
    int topRightIndex = ( mNumRows-1 ) * mNumLines;
    mParticleSystem.particles[ topLeftIndex ]->position = mLeftCorner;
    mParticleSystem.particles[ topRightIndex ]->position = mRightCorner;
    ```

1.  构建并运行应用程序；您将看到一个带有重力下落的粒子网格，其顶部角落被锁定。![如何做到这一点…](img/8703OS_06_05.jpg)

1.  让我们通过允许用户用鼠标拖动粒子来增加一些交互性。声明一个`Particle`指针来存储正在拖动的粒子。

    ```cpp
    Particle *mDragParticle;
    ```

1.  在`setup`方法中初始化粒子为`NULL`。

    ```cpp
    mDragParticle = NULL;
    ```

1.  在应用程序的类声明中声明`mouseUp`和`mouseDown`方法。

    ```cpp
    void mouseDown( MouseEvent event );
    void mouseUp( MouseEvent event );
    ```

1.  在`mouseDown`事件的实现中，我们遍历所有粒子，如果粒子位于光标下，我们将`mDragParticle`设置为指向它。

    ```cpp
    void MyApp::mouseDown( MouseEvent event ){
    for( vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ){
            Particle *part = *it;
            float dist = part->position.distance( event.getPos() );
    if( dist< part->radius ){
    mDragParticle = part;
    return;
            }
        }
    }
    ```

1.  在`mouseUp`事件中，我们只需将`mDragParticle`设置为`NULL`。

    ```cpp
    void MyApp::mouseUp( MouseEvent event ){
    mDragParticle = NULL;
    }
    ```

1.  我们需要检查`mDragParticle`是否是一个有效的指针并将粒子的位置设置为鼠标光标。将以下代码添加到`update`方法中：

    ```cpp
    if( mDragParticle != NULL ){
    mDragParticle->position = getMousePos();
        }
    ```

1.  构建并运行应用程序。按住并拖动鼠标到任何粒子上，然后将其拖动以查看布料模拟如何反应。

## 它是如何工作的…

布料模拟是通过创建一个二维粒子网格并使用弹簧连接它们来实现的。每个粒子将与其相邻的粒子以及其上方和下方的粒子通过弹簧连接。

## 更多内容…

网格的密度可以根据用户的需求进行更改。使用具有更多粒子的网格将生成更精确的模拟，但速度会较慢。

通过更改`mNumLines`和`mNumRows`来更改构成网格的粒子的数量。

# 布料模拟纹理化

在这个菜谱中，我们将学习如何将纹理应用到我们在当前章节的*创建布料模拟*菜谱中创建的布料模拟上。

## 准备工作

我们将使用在菜谱*创建布料模拟*中开发的布料模拟作为这个菜谱的基础。

您还需要一个图像作为纹理；将其放置在您的`assets`文件夹中。在这个菜谱中，我们将我们的图像命名为`texture.jpg`。

## 如何做到这一点…

我们将计算布料模拟中每个粒子的对应纹理坐标并应用纹理。

1.  包含必要的文件以使用纹理和读取图像。

    ```cpp
    #include "cinder/gl/Texture.h"
    #include "cinder/ImageIo.h"
    ```

1.  在你的应用程序类声明中声明一个`ci::gl::Texture`对象。

    ```cpp
    gl::Texture mTexture;
    ```

1.  在`setup`方法中加载图像。

    ```cpp
    mTexture = loadImage( loadAsset( "image.jpg" ) );
    ```

1.  我们将重新制作`draw`方法。所以我们将擦除在*创建布料模拟*配方中更改的所有内容，并应用`clear`方法。你的`draw`方法应该如下所示：

    ```cpp
    void MyApp::draw(){
      gl::clear( Color( 0, 0, 0 ) );
    }
    ```

1.  在调用`clear`方法之后，启用`VERTEX`和`TEXTURE COORD`数组并绑定纹理。将以下代码添加到`draw`方法中：

    ```cpp
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_TEXTURE_COORD_ARRAY );
    mTexture.enableAndBind();
    ```

1.  现在，我们将遍历构成布料模拟网格的所有粒子和弹簧，并在每一行及其相邻行之间绘制一个纹理三角形条带。首先创建一个`for`循环，循环`mNumRows-1`次，并创建两个`std::vector<Vec2f>`容器来存储顶点和纹理坐标。

    ```cpp
    for( int i=0; i<mNumRows-1; i++ ){
      vector<Vec2f>vertexCoords, textureCoords;
    }
    ```

1.  在循环内部，我们将创建一个嵌套循环，该循环将遍历布料网格中的所有行。在这个循环中，我们将计算将要绘制的粒子的索引，计算它们对应的纹理坐标，并将它们与`textureCoords`和`vertexCoords`的位置一起添加。将以下代码输入到我们在上一步创建的循环中：

    ```cpp
    or( int j=0; j<mNumLines; j++ ){
     int indexTopLeft = i * mNumLines + j;
     int indexTopRight = ( i+1) * mNumLines + j;
     Particle *left = mParticleSystem.particles[ indexTopLeft ];
     Particle *right = mParticleSystem.particles[indexTopRight ];
     float texX = ( (float)i / (float)(mNumRows-1) ) * mTexture.getRight();
     float texY = ( (float)j / (float)(mNumLines-1) ) * mTexture.getBottom();
     textureCoords.push_back( Vec2f( texX, texY ) );
     vertexCoords.push_back( left->position );
     texX = ( (float)(i+1) / (float)(mNumRows-1) ) * mTexture.getRight();
     textureCoords.push_back( Vec2f( texX, texY ) );
     vertexCoords.push_back( right->position );
    }
    ```

    现在已经计算并放置了`vertex`和`texture`坐标到`vertexCoords`和`textureCoords`中，我们将绘制它们。以下是完整的嵌套循环：

    ```cpp
    for( int i=0; i<mNumRows-1; i++ ){
     vector<Vec2f> vertexCoords, textureCoords;
     for( int j=0; j<mNumLines; j++ ){
      int indexTopLeft = i * mNumLines + j;
      int indexTopRight = ( i+1) * mNumLines + j;
      Particle *left = mParticleSystem.particles[ indexTopLeft ];
      Particle *right = mParticleSystem.particles[ indexTopRight ];
      float texX = ( (float)i / (float)(mNumRows-1) ) * mTexture.getRight();
      float texY = ( (float)j / (float)(mNumLines-1) ) * mTexture.getBottom();
      textureCoords.push_back( Vec2f( texX, texY ) );
      vertexCoords.push_back( left->position );
      texX = ( (float)(i+1) / (float)(mNumRows-1) ) * mTexture.getRight();
      textureCoords.push_back( Vec2f( texX, texY ) );
      vertexCoords.push_back( right->position );
     }
     glVertexPointer 2, GL_FLOAT, 0, &vertexCoords[0] );
     glTexCoordPointer( 2, GL_FLOAT, 0, &textureCoords[0] );
     glDrawArrays( GL_TRIANGLE_STRIP, 0, vertexCoords.size() );
    }
    ```

1.  最后，我们需要通过添加以下代码来解除`mTexture`的绑定：

    ```cpp
    mTexture.unbind();
    ```

## 它是如何工作的…

我们根据粒子在网格上的位置计算了相应的纹理坐标。然后，我们绘制了由行上的粒子及其相邻行上的粒子形成的三角形条带纹理。

# 使用点精灵和着色器纹理化粒子系统

在这个配方中，我们将学习如何使用 OpenGL 点精灵和 GLSL 着色器将纹理应用到所有粒子。

此方法经过优化，允许以快速帧率绘制大量粒子。

## 准备工作

我们将使用在第五章中开发的 2D 粒子系统配方*在 2D 中创建粒子系统*，所以我们需要将以下文件添加到你的项目中：

+   `Particle.h`

+   `ParticleSystem.h`

我们还将加载一个图像作为纹理使用。图像的大小必须是 2 的幂，例如 256 x 256 或 512 x 512。将图像放置在`assets`文件夹中，并命名为`particle.png`。

## 如何做到这一点...

我们将创建一个 GLSL 着色器，然后启用 OpenGL 点精灵来绘制纹理化的粒子。

1.  让我们从创建 GLSL 着色器开始。创建以下文件：

    +   `shader.frag`

    +   `shader.vert`

    将它们添加到`assets`文件夹中。

1.  在你选择的 IDE 中打开文件`shader.frag`并声明一个`uniform sampler2D`：

    ```cpp
    uniform sampler2D tex; 
    ```

1.  在`main`函数中，我们使用纹理来定义片段颜色。添加以下代码：

    ```cpp
    void main (void) {
      gl_FragColor = texture2D(tex, gl_TexCoord[0].st) * gl_Color;
    }
    ```

1.  打开`shader.vert`文件并创建一个`float attribute`来存储粒子的半径。在`main`方法中，我们定义了位置、颜色和点大小属性。添加以下代码：

    ```cpp
    attribute float particleRadius;
    void main(void)
    {
      gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
      gl_PointSize = particleRadius;
      gl_FrontColor = gl_Color;
    }
    ```

1.  我们的着色器完成了！让我们转到我们的应用程序源文件并包含必要的文件。将以下代码添加到您的应用程序源文件中：

    ```cpp
    #include "cinder/gl/Texture.h"
    #include "cinder/ImageIo.h"
    #include "cinder/Rand.h"
    #include "cinder/gl/GlslProg.h"
    #include "ParticleSystem.h"
    ```

1.  声明创建粒子系统的成员变量以及存储粒子位置和半径的数组。还声明一个变量来存储粒子的数量。

    ```cpp
    ParticleSystem mParticleSystem;
    int mNumParticles;
    Vec2f *mPositions;
    float *mRadiuses;
    ```

1.  在`setup`方法中，我们将`mNumParticles`初始化为`1000`并分配数组。我们还将创建随机粒子。

    ```cpp
    mNumParticles = 1000;
    mPositions = new Vec2f[ mNumParticles ];
    mRadiuses = new float[ mNumParticles ];

    for( int i=0; i<mNumParticles; i++ ){
     float x = randFloat( 0.0f, getWindowWidth() );
     float y = randFloat( 0.0f, getWindowHeight() );
     float radius = randFloat( 5.0f, 50.0f );
     Particle *particle = new Particle( Vec2f( x, y ), radius, 1.0f, 0.9f );
     mParticleSystem.addParticle( particle );
    }
    mParticleSystem.addParticle( particle );
    ```

1.  在`update`方法中，我们将更新`mParticleSystem`和`mPositions`以及`mRadiuses`数组。将以下代码添加到`update`方法中：

    ```cpp
    mParticleSystem.update();
    for( int i=0; i<mNumParticles; i++ ){
     mPositions[i] = mParticleSystem.particles[i]->position;
     mRadiuses[i] = mParticleSystem.particles[i]->radius;
    }
    ```

1.  声明着色器和粒子的纹理。

    ```cpp
    gl::Texture mTexture;
    gl::GlslProg mShader;
    ```

1.  通过在`setup`方法中添加以下代码来加载着色器和纹理：

    ```cpp
    mTexture = loadImage( loadAsset( "particle.png" ) );
    mShader = gl::GlslProg( loadAsset( "shader.vert"), loadAsset( "shader.frag" ) );
    ```

1.  在`draw`方法中，我们将首先用黑色清除背景，设置窗口的矩阵，启用加法混合，并绑定着色器。

    ```cpp
    gl::clear( Color( 0, 0, 0 ) ); 
    gl::setMatricesWindow( getWindowWidth(), getWindowHeight() );
    gl::enableAdditiveBlending();
    mShader.bind();
    ```

1.  在`Vertex`着色器中获取`particleRadius`属性的定位。启用顶点属性数组并将指针设置为`mRadiuses`。

    ```cpp
    GLint particleRadiusLocation = mShader.getAttribLocation( "particleRadius" );
    glEnableVertexAttribArray(particleRadiusLocation);
    glVertexAttribPointer(particleRadiusLocation, 1, GL_FLOAT, false, 0, mRadiuses);
    ```

1.  启用点精灵并启用我们的着色器以写入点大小。

    ```cpp
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);	
    ```

1.  启用顶点数组和设置顶点指针为`mPositions`。

    ```cpp
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, mPositions);
    ```

1.  现在启用并绑定纹理，以点形式绘制顶点数组，然后解绑纹理。

    ```cpp
    mTexture.enableAndBind();
    glDrawArrays( GL_POINTS, 0, mNumParticles );
    mTexture.unbind();
    ```

1.  现在我们需要做的就是禁用顶点数组，禁用顶点属性数组，并解绑着色器。

    ```cpp
    glDisableClientState(GL_VERTEX_ARRAY); 
    glDisableVertexAttribArrayARB(particleRadiusLocation);
    mShader.unbind();
    ```

1.  构建并运行应用程序，你将看到应用了纹理的`1000`个随机粒子。![如何操作...](img/8703OS_06_06.jpg)

## 它是如何工作的...

点精灵是 OpenGL 的一个不错特性，允许将整个纹理应用于单个点。当绘制粒子系统时非常实用，并且相当优化，因为它减少了发送到图形卡的信息量，并在 GPU 上执行了大部分计算。

在配方中，我们还创建了一个 GLSL 着色器，一种高级编程语言，它允许对编程管道有更多的控制，为每个粒子定义单独的点大小。

在`update`方法中，我们更新了`Positions`和`Radiuses`数组，因此如果粒子被动画化，数组将代表正确的值。

## 还有更多...

点精灵允许我们在 3D 空间中对点进行纹理化。要绘制 3D 粒子系统，请执行以下操作：

1.  使用配方中*There's more…*部分描述的*在 2D 中创建粒子系统*的`Particle`类，来自第五章，*构建粒子系统*。

1.  声明并初始化`mPositions`为一个`ci::Vec3f`数组。

1.  在`draw`方法中，通过应用以下更改来指示顶点指针包含 3D 信息：

    ```cpp
    glVertexPointer(2, GL_FLOAT, 0, mPositions);
    ```

    将之前的代码行更改为：

    ```cpp
    glVertexPointer(3, GL_FLOAT, 0, mPositions);
    ```

1.  顶点着色器需要根据粒子的深度调整点的大小。`shader.vert` 文件需要读取以下代码：

    ```cpp
    attribute float particleRadius;

    void main(void)
    {
      vec4eyeCoord = gl_ModelViewMatrix * gl_Vertex;
      gl_Position = gl_ProjectionMatrix * eyeCoord;
      float distance = sqrt(eyeCoord.x*eyeCoord.x + eyeCoord.y*eyeCoord.y + eyeCoord.z*eyeCoord.z);
      float attenuation = 3000.0 / distance;
      gl_PointSize = particleRadius * attenuation;
      gl_FrontColor = gl_Color;
    }
    ```

# 连接点

在这个菜谱中，我们将展示如何用线条连接粒子，并介绍另一种绘制粒子的方法。

## 开始

本菜谱的代码库是来自菜谱 *模拟粒子随风飘动* 的示例（来自第五章，*构建粒子系统*），请参考此菜谱。

## 如何做到这一点…

我们将使用线条连接渲染为圆形的粒子。

1.  在 `setup` 方法中更改要创建的粒子数量：

    ```cpp
    int numParticle = 100;
    ```

1.  我们将按照以下方式计算每个粒子的 `radius` 和 `mass`：

    ```cpp
    float radius = Rand::randFloat(2.f, 5.f);
    float mass = radius*2.f;
    ```

1.  将 `Particle.cpp` 源文件内的 `draw` 方法替换为以下内容：

    ```cpp
    void Particle::draw(){
     ci::gl::drawSolidCircle(position, radius);
     ci::gl::drawStrokedCircle(position, radius+2.f);
    }
    ```

1.  按如下方式替换 `ParticleSystem.cpp` 源文件内的 `draw` 方法：

    ```cpp
    void ParticleSystem::draw(){
     gl::enableAlphaBlending();
     std::vector<Particle*>::iterator it;
     for(it = particles.begin(); it != particles.end(); ++it){
      std::vector<Particle*>::iterator it2;
      for(it2=particles.begin(); it2!= particles.end(); ++it2){
       float distance = (*it)->position.distance( 
        (*it2)->position ));
       float per = 1.f - (distance / 100.f);
       ci::gl::color( ci::ColorA(1.f,1.f,1.f, per*0.8f) );
       ci::Vec2f conVec = (*it2)->position-(*it)->position;
       conVec.normalize();
       ci::gl::drawLine(
        (*it)->position+conVec * ((*it)->radius+2.f),
        (*it2)->position-conVec * ((*it2)->radius+2.f ));
      }
     }
     ci::gl::color( ci::ColorA(1.f,1.f,1.f, 0.8f) );
     std::vector<Particle*>::iterator it3;
     for(it3 = particles.begin(); it3!= particles.end(); ++it3){
      (*it3)->draw();
     }
    }
    ```

## 如何工作…

本示例中最有趣的部分在第 4 步中提到。我们正在遍历所有点，实际上是所有可能的点对，用线条连接它们并应用适当的透明度。连接两个粒子的线条透明度是根据这两个粒子之间的距离计算的；距离越长，连接线越透明。

![如何工作…](img/8703OS_06_07.jpg)

看看第 3 步中粒子的绘制方式。它们是带有略微更大的外圆的实心圆。一个很好的细节是我们绘制粒子之间的连接线，这些粒子粘附在外圆的边缘，但不会穿过它。我们在第 4 步中做到了这一点，当时我们计算了连接两个粒子的向量的归一化向量，然后使用它们将连接点移动到该向量，乘以外圆半径。

![如何工作…](img/8703OS_06_08.jpg)

# 使用样条曲线连接粒子

在这个菜谱中，我们将学习如何在 3D 中用样条曲线连接粒子。

## 开始

在这个菜谱中，我们将使用来自菜谱 *创建粒子系统* 的粒子代码库，该菜谱位于第五章，*构建粒子系统*。我们将使用 3D 版本。

## 如何做到这一点…

我们将创建连接粒子的样条曲线。

1.  在 `ParticleSystem.h` 内包含必要的头文件：

    ```cpp
    #include "cinder/BSpline.h"
    ```

1.  向 `ParticleSystem` 类添加一个新属性：

    ```cpp
    ci::BSpline3f spline;
    ```

1.  为 `ParticleSystem` 类实现 `computeBSpline` 方法：

    ```cpp
    void ParticleSystem::computeBspline(){ 
     std::vector<ci::Vec3f> splinePoints;
     std::vector<Particle*>::iterator it;
     for(it = particles.begin(); it != particles.end(); ++it ){
      ++it;
      splinePoints.push_back( ci::Vec3f( (*it)->position ) );
     }
     spline = ci::BSpline3f( splinePoints, 3, false, false );
    }
    ```

1.  在 `ParticleSystem` 更新方法的末尾调用以下代码：

    ```cpp
    computeBSpline();
    ```

1.  将 `ParticleSystem` 的 `draw` 方法替换为以下内容：

    ```cpp
    void ParticleSystem::draw(){
     ci::gl::color(ci::Color::black());
     if(spline.isUniform()) {
      glBegin(GL_LINES);
      float step = 0.001f;
      for( float t = step; t <1.0f; t += step ) {
       ci::gl::vertex( spline.getPosition( t-step ) );
       ci::gl::vertex( spline.getPosition( t ) );
      } 
      glEnd();
     }
     ci::gl::color(ci::Color(0.0f,0.0f,1.0f));
     std::vector<Particle*>::iterator it;
     for(it = particles.begin(); it != particles.end(); ++it ){
      (*it)->draw();
     }
    }
    ```

1.  为你的主 Cinder 应用程序类文件添加头文件：

    ```cpp
    #include "cinder/app/AppBasic.h"
    #include "cinder/gl/Texture.h"
    #include "cinder/Rand.h"
    #include "cinder/Surface.h"
    #include "cinder/MayaCamUI.h"
    #include "cinder/BSpline.h"

    #include "ParticleSystem.h"
    ```

1.  为你的 `main` 类添加成员：

    ```cpp
    ParticleSystem mParticleSystem;

    float repulsionFactor;
    float maxAlignSpeed;

    CameraPersp        mCam;
    MayaCamUI mMayaCam;

    Vec3f mRepPosition;

    BSpline3f   spline;
    ```

1.  按如下方式实现 `setup` 方法：

    ```cpp
    void MainApp::setup()
    {
    repulsionFactor = -1.0f;
    maxAlignSpeed = 10.f;
    mRepPosition = Vec3f::zero();

    mCam.setPerspective(45.0f, getWindowAspectRatio(), 0.1, 10000);
    mCam.setEyePoint(Vec3f(7.f,7.f,7.f));
    mCam.setCenterOfInterestPoint(Vec3f::zero());
    mMayaCam.setCurrentCam(mCam);
    vector<Vec3f> splinePoints;
    float step = 0.5f;
    float width = 20.f;
    for (float t = 0.f; t < width; t += step) {
     float mass = Rand::randFloat(20.f, 25.f);
     float drag = 0.95f;
     splinePoints.push_back( Vec3f(math<float>::cos(t),
     math<float>::sin(t),
     t - width*0.5f) );
     Particle *particle;
     particle = new Particle( 
      Vec3f( math<float>::cos(t)+Rand::randFloat(-0.8f,0.8f),
       math<float>::sin(t)+Rand::randFloat(-0.8f,0.8f),
       t - width*0.5f), 
      1.f, mass, drag );
     mParticleSystem.addParticle( particle );
    }
    spline = BSpline3f( splinePoints, 3, false, false );
    }
    ```

1.  为相机导航添加成员：

    ```cpp
    void MainApp::resize( ResizeEvent event ){
      mCam = mMayaCam.getCamera();
      mCam.setAspectRatio(getWindowAspectRatio());
      mMayaCam.setCurrentCam(mCam);
    }

    void MainApp::mouseDown(MouseEvent event){
      mMayaCam.mouseDown( event.getPos() );
    }

    void MainApp::mouseDrag( MouseEvent event ){
      mMayaCam.mouseDrag( event.getPos(), event.isLeftDown(), event.isMiddleDown(), event.isRightDown() );
    }
    ```

1.  按如下方式实现 `update` 方法：

    ```cpp
    void MainApp::update() {
     float pos=math<float>::abs(sin(getElapsedSeconds()*0.5f));
     mRepPosition = spline.getPosition( pos );
     std::vector<Particle*>::iterator it;
     it = mParticleSystem.particles.begin();
     for(; it != mParticleSystem.particles.end(); ++it ) {
      Vec3f repulsionForce = (*it)->position - mRepPosition;
      repulsionForce = repulsionForce.normalized() *
       math<float>::max(0.f, 3.f-repulsionForce.length());
      (*it)->forces += repulsionForce;
      Vec3f alignForce = (*it)->anchor - (*it)->position;
      alignForce.limit(maxAlignSpeed);
      (*it)->forces += alignForce;
     }
     mParticleSystem.update();
    }
    ```

1.  按如下方式实现 `draw` 方法：

    ```cpp
    void MainApp::draw() {
     gl::enableDepthRead();
     gl::enableDepthWrite();
     gl::clear( Color::white() );
     gl::setViewport(getWindowBounds());
     gl::setMatrices(mMayaCam.getCamera());
     gl::color(Color(1.f,0.f,0.f));
     gl::drawSphere(mRepPosition, 0.25f);
     mParticleSystem.draw();
    }
    ```

## 如何工作…

**B 样条**使我们能够通过一些给定的点绘制一条非常平滑的曲线，在我们的情况下，是粒子位置。我们仍然可以应用一些吸引和排斥力，使得这条线的行为相当像弹簧。在 Cinder 中，你可以在 2D 和 3D 空间中使用 B 样条，并使用 `BSpline` 类来计算它们。

![工作原理…](img/8703OS_06_09.jpg)

## 参考信息

关于 B 样条的更多详细信息可在 [`zh.wikipedia.org/wiki/B 样条`](http://zh.wikipedia.org/wiki/B 样条) 找到。
