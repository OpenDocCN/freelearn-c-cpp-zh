# 第五章：构建粒子系统

在本章中，我们将涵盖：

+   在 2D 中创建粒子系统

+   应用排斥力和吸引力

+   模拟粒子随风飘动

+   模拟集群行为

+   使我们的粒子对声音做出反应

+   将粒子与处理后的图像对齐

+   将粒子与网格表面对齐

+   创建弹簧

# 简介

粒子系统是一种计算技术，使用大量小图形对象执行不同类型的模拟，如爆炸、风、火、水和集群。

在本章中，我们将学习如何使用流行的多功能物理算法创建和动画化粒子。

# 在 2D 中创建粒子系统

在这个配方中，我们将学习如何使用 Verlet 算法在二维空间中构建一个基本的粒子系统。

## 准备工作

我们需要创建两个类，一个 `Particle` 类代表单个粒子，一个 `ParticleSystem` 类来管理我们的粒子。

使用您选择的 IDE 创建以下文件：

+   `Particle.h`

+   `Particle.cpp`

+   `ParticleSystem.h`

+   `ParticleSystem.cpp`

## 如何做呢...

我们将学习如何创建一个基本的粒子系统。执行以下步骤：

1.  首先，让我们在 `Particle.h` 文件中声明我们的 `Particle` 类并包含必要的 Cinder 文件：

    ```cpp
    #pragma once

    #include "cinder/gl/gl.h"
    #include "cinder/Vector.h"

    class Particle{
    };
    ```

1.  让我们在类声明中添加必要的成员变量 - 使用 `ci::Vec2f` 存储位置、前一个位置和施加的力；以及使用 `float` 存储粒子半径、质量和阻力。

    ```cpp
    ci::Vec2f position, prevPosition;
    ci::Vec2f forces;
    float radius;
    float mass;
    float drag;
    ```

1.  为了最终完成 `Particle` 声明，还需要添加一个构造函数，该构造函数接受粒子的初始位置、半径、质量和阻力，以及更新和绘制粒子的方法。

    以下是最终的 `Particle` 类声明：

    ```cpp
    class Particle{
    public:

    Particle( const ci::Vec2f& position, float radius, 
    float mass, float drag );

    void update();
    void draw();

    ci::Vec2f position, prevPosition;
    ci::Vec2f forces;
    float radius;
    float mass;
    float drag;
    };
    ```

1.  让我们继续到 `Particle.cpp` 文件并实现 `Particle` 类。

    第一个必要的步骤是包含 `Particle.h` 文件，如下所示：

    ```cpp
    #include "Particle.h"
    ```

1.  我们将成员变量初始化为构造函数中传递的值。我们还初始化 `forces` 为 `zero` 和 `prevPosition` 为初始位置。

    ```cpp
    Particle::Particle( const ci::Vec2f& position, float radius, float mass, float drag ){
      this->position = position;
      this->radius = radius;
      this->mass = mass;
      this->drag = drag;
      prevPosition = position;
      forces = ci::Vec2f::zero();
    }
    ```

1.  在 `update` 方法中，我们需要创建一个临时的 `ci::Vec2f` 变量来存储更新前的粒子位置。

    ```cpp
    ci::Vec2f temp = position;
    ```

1.  我们通过计算当前位置与前一个位置之间的差异并乘以 `drag` 来计算粒子的速度。为了清晰起见，我们将此值暂时存储在 `ci::Vec2f` 中。

    ```cpp
    ci::Vec2f vel = ( position – prevPosition ) * drag;
    ```

1.  要更新粒子的位置，我们将之前计算的速度加上 `forces` 除以 `mass`。

    ```cpp
    position += vel + forces / mass;
    ```

1.  `update` 方法中的最后一步是将之前存储的位置复制到 `prevPosition` 并将 `forces` 重置为零向量。

    以下是完全的 `update` 方法实现：

    ```cpp
    void Particle::update(){
        ci::Vec2f temp = position;
        ci::Vec2f vel = ( position - prevPosition ) * drag;
        position += vel + forces / mass;
        prevPosition = temp;
        forces = ci::Vec2f::zero();
    }
    ```

1.  在 `draw` 实现中，我们只需在粒子的位置处绘制一个圆，使用其半径。

    ```cpp
    void Particle::draw(){
        ci::gl::drawSolidCircle( position, radius );
    }
    ```

1.  现在随着`Particle`类的完成，我们需要开始着手于`ParticleSystem`类的开发。切换到`ParticleSystem.h`文件，包含必要的文件，并创建`ParticleSystem`类的声明。

    ```cpp
    #pragma once

    #include "Particle.h"
    #include <vector>

    classParticleSystem{
    public:

    };
    ```

1.  让我们添加一个析构函数和更新和绘制粒子的方法。我们还需要创建添加和销毁粒子的方法，以及一个`std::vector`变量来存储系统中的粒子。以下是最终的类声明：

    ```cpp
    Class ParticleSystem{
    public:
      ~ParticleSystem();

      void update();
      void draw();

      void addParticle( Particle *particle );
      void destroyParticle( Particle *particle );

        std::vector<Particle*> particles;

    };
    ```

1.  切换到`ParticleSystem.cpp`文件，让我们开始实现。首先我们需要做的是包含包含类声明的文件。

    ```cpp
    #include "ParticleSystem.h"
    ```

1.  现在，让我们逐一实现这些方法。在析构函数中，我们遍历所有粒子并将它们删除。

    ```cpp
    ParticleSystem::~ParticleSystem(){
      for( std::vector<Particle*>::iterator it = particles.begin(); it!= particles.end(); ++it ){
      delete *it;
        }
      particles.clear();
    }
    ```

1.  `update`方法将用于遍历所有粒子并对每个粒子调用`update`。

    ```cpp
    void ParticleSystem::update(){
      for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ){
            (*it)->update();
        }
    }
    ```

1.  `draw`方法将遍历所有粒子，并对每个粒子调用`draw`。

    ```cpp
    void ParticleSystem::draw(){
      for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ){
            (*it)->draw();
        }
    }
    ```

1.  `addParticle`方法将粒子插入到`particles`容器中。

    ```cpp
    void ParticleSystem::addParticle( Particle *particle ){
      particles.push_back( particle );
    }
    ```

1.  最后，`destroyParticle`将删除粒子并从粒子列表中移除。

    我们将找到粒子的迭代器，并使用它来从容器中删除和稍后移除对象。

    ```cpp
    void ParticleSystem::destroyParticle( Particle *particle ){
      std::vector<Particle*>::iterator it = std::find( particles.begin(), particles.end(), particle );
      delete *it;
      particles.erase( it );
    }
    ```

1.  在我们的类准备就绪后，让我们转到应用程序类并创建一些粒子。

    在我们的应用程序类中，我们需要在源文件顶部包含`ParticleSystem`头文件和必要的头文件以使用随机数：

    ```cpp
    #include "ParticleSystem.h"
    #include "cinder/Rand.h"
    ```

1.  在类声明中声明一个`ParticleSystem`对象。

    ```cpp
    ParticleSystem mParticleSystem;
    ```

1.  在`setup`方法中，我们可以在窗口上创建 100 个具有随机位置和随机半径的粒子。我们将质量定义为与半径相同，以此作为大小和质量的关联方式。`drag`将被设置为 9.5。

    在`setup`方法内部添加以下代码片段：

    ```cpp
    int numParticle = 100;
      for( int i=0; i<numParticle; i++ ){
      float x = ci::randFloat( 0.0f, getWindowWidth() );
      float y = ci::randFloat( 0.0f, getWindowHeight() );
      float radius = ci::randFloat( 5.0f, 15.0f );
      float mass = radius;radius;
      float drag = 0.95f;
            Particle *particle = new Particle
            ( Vec2f( x, y ), radius, mass, drag );
            mParticleSystem.addParticle( particle );
    }
    ```

1.  在`update`方法中，我们需要通过在`mParticleSystem`上调用`update`方法来更新粒子。

    ```cpp
    void MyApp::update(){
      mParticleSystem.update();
    }
    ```

1.  在`draw`方法中，我们需要清除屏幕，设置窗口的矩阵，并在`mParticleSystem`上调用`draw`方法。

    ```cpp
    void ParticlesApp::draw()
    {
      gl::clear( Color( 0, 0, 0 ) ); 
      gl::setMatricesWindow( getWindowWidth(), getWindowHeight() );
      mParticleSystem.draw();
    }
    ```

1.  构建并运行应用程序，你将在屏幕上看到 100 个随机圆圈，如下面的截图所示：![如何做到这一点…](img/8703OS_5_1.jpg)

在接下来的菜谱中，我们将学习如何以有机和吸引人的方式动画化粒子。

## 它是如何工作的...

之前描述的方法使用了一个流行且通用的 Verlet 积分器。其主要特点是对速度的隐式近似。这是通过在每次模拟更新时计算自上次模拟更新以来所经过的距离来实现的。这允许有更高的稳定性，因为速度是隐式地与位置相关联的，并且不太可能发生不同步。

`drag` 成员变量代表运动阻力，应该是一个介于 0.0 和 1.0 之间的数字。0.0 的值表示如此大的阻力，以至于粒子将无法移动。1.0 的值表示没有阻力，将使粒子无限期地移动。我们在第 7 步中应用了 `drag`，其中我们将 `drag` 乘以速度：

```cpp
ci::Vec2f vel = ( position – prevPosition ) * drag;
```

## 还有更多…

要在 3D 中创建粒子系统，必须使用 3D 向量而不是 2D 向量。

由于 Cinder 的 2D 向量和 3D 向量类具有非常相似的类结构，我们只需将 `position`、`prevPosition` 和 `forces` 改为 `ci::Vec3f` 对象。

构造函数也需要接受一个 `ci::Vec3f` 对象作为参数。

以下是根据这些更改的类声明：

```cpp
class Particle{
public:

    Particle( const ci::Vec3f& position, 
    float radius, float mass, float drag );

    void update();
    void draw();

    ci::Vec3f position, prevPosition;
    ci::Vec3f forces;
    float radius;
    float mass;
    float drag;
};
```

`draw` 方法也应更改以允许 3D 绘制；例如，我们可以绘制一个球体而不是圆形：

```cpp
void Particle::draw(){
  ci::gl::drawSphere( position, radius );
} 
```

## 参见

+   关于 Verlet 算法的实现，请参阅托马斯·雅各布森的论文，位于 [`www.pagines.ma1.upc.edu/~susin/contingut/AdvancedCharacterPhysics.pdf`](http://www.pagines.ma1.upc.edu/~susin/contingut/AdvancedCharacterPhysics.pdf)

+   关于 Verlet 积分的更多信息，请阅读维基百科 [`en.wikipedia.org/wiki/Verlet_integration`](http://en.wikipedia.org/wiki/Verlet_integration)。

# 应用排斥力和吸引力

在这个菜谱中，我们将展示如何将排斥力和吸引力应用到我们在前一个菜谱中实现的粒子系统中。

## 准备工作

在这个菜谱中，我们将使用 *Creating particle system in 2D* 菜谱中的代码。

## 如何做到这一点…

我们将展示如何将力应用到粒子系统中。执行以下步骤：

1.  向你的应用程序的主类添加属性。

    ```cpp
    Vec2f attrPosition;
    float attrFactor, repulsionFactor, repulsionRadius;
    ```

1.  在 `setup` 方法中设置默认值。

    ```cpp
    attrPosition = getWindowCenter();
    attrFactor = 0.05f;
    repulsionRadius = 100.f;
    repulsionFactor = -5.f;
    ```

1.  实现以下 `mouseMove` 和 `mouseDown` 方法：

    ```cpp
    void MainApp::mouseMove(MouseEvent event)
    {
      attrPosition.x = event.getPos().x;
      attrPosition.y = event.getPos().y;
    }

    void MainApp::mouseDown(MouseEvent event)
    {
    for( std::vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ) {
      Vec2f repulsionForce = (*it)->position - event.getPos();
      repulsionForce = repulsionForce.normalized() * math<float>::max(0.f, repulsionRadius - repulsionForce.length());
              (*it)->forces += repulsionForce;
          }
    }
    ```

1.  在`update`方法的开始处，添加以下代码片段：

    ```cpp
    for( std::vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ) {
      Vec2f attrForce = attrPosition - (*it)->position;
      attrForce *= attrFactor;
        (*it)->forces += attrForce;
    }
    ```

## 它是如何工作的…

在这个例子中，我们为第一个菜谱中引入的粒子引擎添加了交互。吸引力指向你的鼠标光标位置，而排斥向量指向相反方向。这些力在第 3 和 4 步中计算并应用到每个粒子上，然后我们让粒子跟随你的鼠标光标，但是当你点击左键时，它们会突然远离鼠标光标。这种效果可以通过基本的向量运算实现。Cinder 允许你以与通常对标量进行操作相同的方式执行向量计算。

排斥力在第 3 步计算。我们使用从鼠标光标位置到粒子位置的归一化向量，乘以基于粒子与鼠标光标位置之间的距离计算的排斥因子。使用 `repulsionRadius` 值，我们可以限制排斥力的范围。

我们在第 4 步计算吸引力，取从粒子位置开始到鼠标光标位置的向量。我们将此向量乘以`attrFactor`值，该值控制吸引力的强度。

![如何工作…](img/8703OS_5_2.jpg)

# 模拟风中飞行的粒子

在这个配方中，我们将解释如何将布朗运动应用于您的粒子。粒子将表现得像雪花或随风飘动的树叶。

## 准备工作

在这个配方中，我们将使用*在 2D 中创建粒子系统*配方的代码库。

## 如何实现它...

我们将添加来自 Perlin 噪声和正弦函数计算的粒子运动。执行以下步骤来完成此操作：

1.  添加必要的头文件。

    ```cpp
    #include "cinder/Perlin.h"
    ```

1.  向应用程序的主类添加属性。

    ```cpp
    float    mFrequency;
    Perlin    mPerlin;
    ```

1.  在`setup`方法中设置默认值。

    ```cpp
    mFrequency = 0.01f;
    mPerlin = Perlin();
    ```

1.  改变粒子的数量、半径和质量。

    ```cpp
    int numParticle = 300;
    float radius = 1.f;
    float mass = Rand::randFloat(1.f, 5.f);
    ```

1.  在`update`方法的开头添加以下代码片段：

    ```cpp
    Vec2f oscilationVec;
    oscilationVec.x = sin(getElapsedSeconds()*0.6f)*0.2f;
    oscilationVec.y = sin(getElapsedSeconds()*0.2f)*0.1f;
    std::vector<Particle*>::iterator it;
    for(it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ) {
      Vec2f windForce = mPerlin.dfBm( (*it)->position * mFrequency );
        (*it)->forces += windForce * 0.1f;
        (*it)->forces += oscilationVec;
    }
    ```

## 如何工作…

主要的运动计算和力在第 5 步应用。如您所见，我们正在使用作为 Cinder 一部分实现的 Perlin 噪声算法。它为每个粒子提供检索布朗运动向量的方法。我们还添加了`oscilationVec`，使粒子从左到右和向后摆动，增加更真实的行为。

![如何工作…](img/8703OS_5_3.jpg)

## 参见

+   **Perlin 噪声原始来源**: [`mrl.nyu.edu/~perlin/doc/oscar.html#noise`](http://mrl.nyu.edu/~perlin/doc/oscar.html#noise)

+   **布朗运动**: [`en.wikipedia.org/wiki/Brownian_motion`](http://en.wikipedia.org/wiki/Brownian_motion)

# 模拟群聚行为

群聚是应用于组织成鸟群或其他飞行动物的行为的术语。

从我们的角度来看，特别有趣的是，通过仅对每个粒子（Boid）应用三条规则就可以模拟群聚行为。这些规则如下：

+   **分离**: 避免过于靠近的邻居

+   **对齐**: 驶向邻居的平均速度

+   **聚合**: 驶向邻居的平均位置

## 准备工作

在这个配方中，我们将使用来自*在 2D 中创建粒子系统*配方的代码。

## 如何实现它…

我们将实现群聚行为的规则。执行以下步骤来完成此操作：

1.  改变粒子的数量、半径和质量。

    ```cpp
    int numParticle = 50;
    float radius = 5.f;
    float mass = 1.f;
    ```

1.  在`Particle.h`头文件中为`Particle`类添加新方法和属性的定义。

    ```cpp
    void flock(std::vector<Particle*>& particles);
    ci::Vec2f steer(ci::Vec2f target, bool slowdown);
    void borders(float width, float height);
    ci::Vec2f separate(std::vector<Particle*>& particles);
    ci::Vec2f align(std::vector<Particle*>& particles);
    ci::Vec2f cohesion(std::vector<Particle*>& particles);

    float maxspeed;
    float maxforce;
    ci::Vec2f vel;
    ```

1.  在`Particle.cpp`源文件中的`Particle`构造函数末尾设置`maxspeed`和`maxforce`的默认值。

    ```cpp
    this->maxspeed = 3.f;
    this->maxforce = 0.05f;
    ```

1.  在`Particle.cpp`源文件中实现`Particle`类的新方法。

    ```cpp
    void Particle::flock(std::vector<Particle*>& particles) {
      ci::Vec2f acc;
      acc += separate(particles) * 1.5f;
      acc += align(particles) * 1.0f;
      acc += cohesion(particles) * 1.0f;
      vel += acc;
      vel.limit(maxspeed);
    }

    ci::Vec2f Particle::steer(ci::Vec2f target, bool slowdown) {
    ci::Vec2f steer;
    ci::Vec2f desired = target - position;
    float d = desired.length();
    if (d >0) {
      desired.normalize();
      if ((slowdown) && (d <100.0)) desired *= (maxspeed*(d/100.0));
      else desired *= maxspeed;
      steer = desired - vel;
      steer.limit(maxforce);
        }
    else {
      steer = ci::Vec2f::zero();
        }
      return steer;
    }

    void Particle::borders(float width, float height) {
      if (position.x< -radius) position.x = width+radius;
      if (position.y< -radius) position.y = height+radius;
      if (position.x>width+radius) position.x = -radius;
      if (position.y>height+radius) position.y = -radius;
    }
    ```

1.  添加分离规则的方法。

    ```cpp
    ci::Vec2f Particle::separate(std::vector<Particle*>& particles) {
    ci::Vec2f resultVec = ci::Vec2f::zero();
    float targetSeparation = 30.f;
    int count = 0;
    for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ) {
      ci::Vec2f diffVec = position - (*it)->position;
      if( diffVec.length() >0&&diffVec.length() <targetSeparation ) {
        resultVec += diffVec.normalized() / diffVec.length();
        count++;
            }
        }

    if (count >0) {
      resultVec /= (float)count;
        }

    if (resultVec.length() >0) {
      resultVec.normalize();
      resultVec *= maxspeed;
      resultVec -= vel;
      resultVec.limit(maxforce);
        }

    return resultVec;
    }
    ```

1.  添加对齐规则的方法。

    ```cpp
    ci::Vec2f Particle::align(std::vector<Particle*>& particles) {
    ci::Vec2f resultVec = ci::Vec2f::zero();
    float neighborDist = 50.f;
    int count = 0;
    for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ) {
    ci::Vec2f diffVec = position - (*it)->position;
    if( diffVec.length() >0 && diffVec.length() <neighborDist ) {
    resultVec += (*it)->vel;
    count++;
            }
        }

    if (count >0) {
      resultVec /= (float)count;
    }

      if (resultVec.length() >0) {
      resultVec.normalize();
      resultVec *= maxspeed;
      resultVec -= vel;
      resultVec.limit(maxforce);
        }

      return resultVec;
    }
    ```

1.  添加聚合规则的方法。

    ```cpp
    ci::Vec2f Particle::cohesion(std::vector<Particle*>& particles) {
    ci::Vec2f resultVec = ci::Vec2f::zero();
    float neighborDist = 50.f;
    int count = 0;
    for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ) {
      float d = position.distance( (*it)->position );
      if( d >0 && d <neighborDist ) {
        resultVec += (*it)->position;
        count++;
            }
        }

    if (count >0) {
      resultVec /= (float)count;
      return steer(resultVec, false);
        }

      return resultVec;
    }
    ```

1.  将`update`方法更改为以下内容

    ```cpp
    void Particle::update(){
      ci::Vec2f temp = position;
      position += vel + forces / mass;
      prevPosition = temp;
      forces = ci::Vec2f::zero();
    }
    ```

1.  改变`Particle`的`drawing`方法，如下所示：

    ```cpp
    void Particle::draw(){
      ci::gl::color(1.f, 1.f, 1.f);
      ci::gl::drawSolidCircle( position, radius );
      ci::gl::color(1.f, 0.f, 0.f);
      ci::gl::drawLine(position,
      position+( position - prevPosition).normalized()*(radius+5.f) );
    }
    ```

1.  在`ParticleSystem.cpp`源文件中更改`ParticleSystem`的`update`方法，如下所示：

    ```cpp
    void ParticleSystem::update(){
      for( std::vector<Particle*>::iterator it = particles.begin(); it!= particles.end(); ++it ){
            (*it)->flock(particles);
            (*it)->update();
            (*it)->borders(640.f, 480.f);
        }
    }
    ```

## 它是如何工作的…

从第 4 步开始实现了三个群聚规则——分离、对齐和凝聚力——并在第 10 步应用于每个粒子。在这一步中，我们还通过重置它们的位置来防止 Boids 超出窗口边界。

![它是如何工作的…](img/8703OS_5_12.jpg)

## 参见

+   **群聚**：[`en.wikipedia.org/wiki/Flocking_(behavior)`](http://en.wikipedia.org/wiki/Flocking_(behavior))

# 使我们的粒子响应用户

在这个菜谱中，我们将基于从音频文件中进行的**快速傅里叶变换**（**FFT**）分析来选择之前的粒子系统并添加动画。

FFT 分析将返回一个表示几个频率窗口振幅的值的列表。我们将每个粒子与一个频率窗口相匹配，并使用其值来动画化每个粒子对其他所有粒子施加的排斥力。

这个例子使用了 Cinder 的 FFT 处理器，它仅在 Mac OS X 上可用。

## 准备工作

我们将使用之前菜谱中开发的相同粒子系统，*在 2D 中创建粒子系统*。创建该菜谱中描述的`Particle`和`ParticleSystem`类，并在应用程序源文件的顶部包含`ParticleSystem.h`文件。

## 如何做到这一点…

使用 FFT 分析中的值来动画化我们的粒子。执行以下步骤来完成：

1.  在应用程序的类中声明一个`ParticleSystem`对象和一个变量来存储我们将创建的粒子数量。

    ```cpp
    ParticleSystem mParticleSystem;
    int mNumParticles;
    ```

1.  在`setup`方法中，我们将创建 256 个随机粒子。粒子的数量将与我们从音频分析中接收到的值的数量相匹配。

    粒子将在窗口的随机位置开始，具有随机的大小和质量。`drag`将设置为`0.9`。

    ```cpp
    mNumParticles = 256;
    for( int i=0; i<mNumParticles; i++ ){
      float x = ci::randFloat( 0.0f, getWindowWidth() );
      float y = ci::randFloat( 0.0f, getWindowHeight() );
      float radius = ci::randFloat( 5.0f, 15.0f );
      float mass = radius;
      float drag = 0.9f;
            Particle *particle = new Particle
            ( Vec2f( x, y ), radius, mass, drag );
    mParticleSystem.addParticle( particle );
    }
    ```

1.  在`update`方法中，我们必须调用粒子系统的`update`方法。

    ```cpp
    void MyApp::update(){
    mParticleSystem.update();
    }
    ```

1.  在`draw`方法中，我们必须清除背景，计算窗口的矩阵，并调用粒子系统的`draw`方法。

    ```cpp
    void MyApp::draw()
    {
      gl::clear( Color( 0, 0, 0 ) ); 
    gl::setMatricesWindow( getWindowWidth(), getWindowHeight() );
    mParticleSystem.draw();
    }
    ```

1.  现在让我们加载并播放一个音频文件。我们首先包括加载、播放和执行 FFT 分析的必要文件。在源文件顶部添加以下代码片段：

    ```cpp
    #include "cinder/audio/Io.h"
    #include "cinder/audio/FftProcessor.h"
    #include "cinder/audio/PcmBuffer.h"
    #include "cinder/audio/Output.h"
    ```

1.  现在声明`ci::audio::TrackRef`，它是一个音频轨道的引用。

    ```cpp
    Audio::TrackRef mAudio;
    ```

1.  在`setup`方法中，我们将打开一个文件对话框，允许用户选择要播放的音频文件。

    如果检索到的路径不为空，我们将使用它来加载并添加一个新的音频轨道。

    ```cpp
    fs::path audioPath = getOpenFilePath();
    if( audioPath.empty() == false ){
      mAudio = audio::Output::addTrack( audio::load( audioPath.string()   ) );
    }
    ```

1.  我们将检查`mAudio`是否成功加载并播放。我们还将启用 PCM 缓冲区和循环。

    ```cpp
    if( mAudio ){
      mAudio->enablePcmBuffering( true );
      mAudio->setLooping( true );
      mAudio->play();
    }
    ```

1.  现在我们已经播放了一个音频文件，我们需要开始动画化粒子。首先，我们需要向窗口中心应用一个弹性力。我们通过迭代所有粒子并添加一个力来完成，这个力是粒子位置与窗口中心位置差值的十分之一。

    将以下代码片段添加到`update`方法中：

    ```cpp
    Vec2f center = getWindowCenter();
    for( vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ){
            Particle *particle = *it;
            Vec2f force = 
            ( center - particle->position ) * 0.1f;
    particle->forces += force;
        }
    ```

1.  现在我们必须计算 FFT 分析。这将在每次更新帧后进行一次。

    声明一个局部变量`std::shared_ptr<float>`，用于存储 FFT 的结果。

    我们将获取`mAudio`的 PCM 缓冲区的引用，并在其左通道上执行 FFT 分析。对`mAudio`及其缓冲区进行测试以检查其有效性是一个好的实践。

    ```cpp
    std::shared_ptr<float>fft;
    if( mAudio ){
      audio::PcmBuffer32fRef pcmBuffer = mAudio->getPcmBuffer();
    if( pcmBuffer ){
        fft = audio::calculateFft( pcmBuffer->getChannelData( audio::CHANNEL_FRONT_LEFT ), mNumParticles );
      }
        }
    ```

1.  我们将使用 FFT 分析的结果来调整每个粒子施加的排斥力。

    将以下代码片段添加到`update`方法中：

    ```cpp
    if( fft ){
    float *values = fft.get();
    for( int i=0; i<mParticleSystem.particles.size()-1; i++ ){
    for( int j=i+1; j<mParticleSystem.particles.size(); j++ ){
      Particle *particleA = 
      mParticleSystem.particles[i];
      Particle *particleB = 
      mParticleSystem.particles[j];
      Vec2f delta = particleA->position - 
      particleB->position;
      float distanceSquared = delta.lengthSquared();
      particleA->forces += ( delta / distanceSquared ) * particleB->mass * values[j] * 0.5f;
      particleB->forces -= ( delta / distanceSquared ) * particleA->mass * values[i] * 0.5f;
    ```

1.  构建并运行应用程序；您将被提示选择一个音频文件。选择它，它将开始播放。粒子将根据音频的频率移动并相互推挤。![如何操作…](img/8703OS_5_6.jpg)

## 它是如何工作的…

我们为 FFT 分析返回的每个值创建了一个粒子，并根据相应的频率窗口幅度使每个粒子排斥其他粒子。随着音乐的演变，动画将相应地做出反应。

## 参见

+   要了解更多关于快速傅里叶变换的信息，请访问[`en.wikipedia.org/wiki/Fast_Fourier_transform`](http://en.wikipedia.org/wiki/Fast_Fourier_transform)

# 将粒子对齐到处理后的图像

在本食谱中，我们将展示如何使用在前面的食谱中介绍的技术使粒子对齐到图像中检测到的边缘。

## 准备工作

在本食谱中，我们将使用来自*在 2D 中创建粒子系统*食谱的粒子实现；来自第三章的*检测边缘*食谱中的图像处理示例；以及*应用排斥和吸引力量*食谱中涵盖的模拟排斥。

## 如何操作…

我们将创建与图像中检测到的边缘对齐的粒子。为此，请执行以下步骤：

1.  在`Particle.h`文件中为`Particle`类添加一个`anchor`属性。

    ```cpp
    ci::Vec2f anchor;
    ```

1.  在`Particle.cpp`源文件的`Particle`类构造函数末尾设置`anchor`值。

    ```cpp
    anchor = position;
    ```

1.  向您应用程序的主类添加一个新属性。

    ```cpp
    float maxAlignSpeed;
    ```

1.  在`setup`方法末尾，在图像处理之后，添加新粒子，如下所示：

    ```cpp
    mMouseDown = false;
    repulsionFactor = -1.f;
    maxAlignSpeed = 10.f;

    mImage = loadImage( loadAsset("image.png") );
    mImageOutput = Surface8u(mImage.getWidth(), mImage.getHeight(), false);

    ip::grayscale(mImage, &mImage);
    ip::edgeDetectSobel(mImage, &mImageOutput);

    Surface8u::Iter pixelIter = mImageOutput.getIter(Area(1,1,mImageOutput.getWidth()-1,mImageOutput.getHeight()-1));

    while( pixelIter.line() ) {
        while( pixelIter.pixel() ) {
            if(pixelIter.getPos().x < mImageOutput.getWidth()
              && pixelIter.getPos().y < 
              mImageOutput.getHeight()
              && pixelIter.r() > 99) {
                float radius = 1.5f;
                float mass = Rand::randFloat(10.f, 20.f);
                float drag = 0.9f;
                Particle *particle = new Particle( 
                pixelIter.getPos(), radius, mass, drag );
                mParticleSystem.addParticle( particle );
            }
        }
    }
    ```

1.  为您的主类实现`update`方法，如下所示：

    ```cpp
    void MainApp::update() {
      for( std::vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ) {

        if(mMouseDown) {
          Vec2f repulsionForce = (*it)->position - getMousePos();
          repulsionForce = repulsionForce.normalized() * math<float>::max(0.f, 100.f - repulsionForce.length());
                      (*it)->forces += repulsionForce;
            }

        Vec2f alignForce = (*it)->anchor - (*it)->position;
        alignForce.limit(maxAlignSpeed);
            (*it)->forces += alignForce;
        }

      mParticleSystem.update();
    }
    ```

1.  将`Particle.cpp`源文件中的`Particle`类的`draw`方法更改为以下内容

    ```cpp
    void Particle::draw(){
      glBegin(GL_POINTS);
      glVertex2f(position);
      glEnd();
    }
    ```

## 它是如何工作的…

第一个主要步骤是在图像的一些特征点上分配粒子。为此，我们检测了边缘，这在第三章的*检测边缘*食谱中有介绍，*使用图像处理技术*。在第 4 步中，您可以看到我们遍历了每个处理图像的每个像素，并在检测到的特征处放置粒子。

你可以在第 5 步找到一个重要的计算，我们尝试将粒子移动回存储在`anchor`属性中的原始位置。为了使粒子无序，我们使用了与*应用排斥和吸引力的力*菜谱中相同的排斥代码。

![如何工作…](img/8703OS_5_8.jpg)

## 参见

+   要了解更多关于快速傅里叶变换的信息，请访问[`en.wikipedia.org/wiki/Fast_Fourier_transform`](http://en.wikipedia.org/wiki/Fast_Fourier_transform)

# 将粒子对齐到网格表面

在这个菜谱中，我们将使用来自*在 2D 中创建粒子系统*菜谱的粒子代码库的 3D 版本。为了在 3D 空间中导航，我们将使用在第二章*为开发做准备*中介绍的*使用 MayaCamUI*菜谱中的`MayaCamUI`。请参阅第二章。

## 准备工作

为了模拟排斥力，我们使用了来自*应用排斥和吸引力的力*菜谱的代码，并对三维空间进行了轻微修改。对于这个例子，我们使用了位于 Cinder 包内 Picking3D 样本的`resources`目录中的`ducky.mesh`网格文件。请将此文件复制到您项目中的`assets`文件夹。

## 如何做…

我们将创建与网格对齐的粒子。执行以下步骤：

1.  在`Particle.h`文件中将`anchor`属性添加到`Particle`类中。

    ```cpp
    ci::Vec3f anchor;
    ```

1.  在`Particle.cpp`源文件的`Particle`类构造函数的末尾设置`anchor`值。

    ```cpp
    anchor = position;
    ```

1.  在您的主类中添加必要的头文件。

    ```cpp
    #include "cinder/TriMesh.h"
    ```

1.  将新属性添加到您应用程序的主类中。

    ```cpp
    ParticleSystem mParticleSystem;

    float repulsionFactor;
    float maxAlignSpeed;

    CameraPersp  mCam;
    MayaCamUI       mMayaCam;

    TriMesh  mMesh;
    Vec3f    mRepPosition;
    ```

1.  在`setup`方法中设置默认值。

    ```cpp
    repulsionFactor = -1.f;
    maxAlignSpeed = 10.f;
    mRepPosition = Vec3f::zero();

    mMesh.read( loadAsset("ducky.msh") );

    mCam.setPerspective(45.0f, getWindowAspectRatio(), 0.1, 10000);
    mCam.setEyePoint(Vec3f(7.f,7.f,7.f));
    mCam.setCenterOfInterestPoint(Vec3f::zero());
    mMayaCam.setCurrentCam(mCam);
    ```

1.  在`setup`方法的末尾添加以下代码片段：

    ```cpp
    for(vector<Vec3f>::iterator it = mMesh.getVertices().begin(); it != mMesh.getVertices().end(); ++it) {
      float mass = Rand::randFloat(2.f, 15.f);
      float drag = 0.95f;
      Particle *particle = new Particle
      ( (*it), 0.f, mass, drag );
      mParticleSystem.addParticle( particle );
    }
    ```

1.  添加相机导航的方法。

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
      mMayaCam.mouseDrag( event.getPos(), event.isLeftDown(), 
      event.isMiddleDown(), event.isRightDown() );
    }
    ```

1.  为您的应用程序主类实现`update`和`draw`方法。

    ```cpp
    void MainApp::update() {

    mRepPosition.x = cos(getElapsedSeconds()) * 3.f;
    mRepPosition.y = sin(getElapsedSeconds()*2.f) * 3.f;
    mRepPosition.z = cos(getElapsedSeconds()*1.5f) * 3.f;

    for( std::vector<Particle*>::iterator it = mParticleSystem.particles.begin(); it != mParticleSystem.particles.end(); ++it ) {

      Vec3f repulsionForce = (*it)->position - mRepPosition;
      repulsionForce = repulsionForce.normalized() * math<float>::max(0.f, 3.f - repulsionForce.length());
      (*it)->forces += repulsionForce;

      Vec3f alignForce = (*it)->anchor - (*it)->position;
      alignForce.limit(maxAlignSpeed);
            (*it)->forces += alignForce;
        }

      mParticleSystem.update();
    }

    void MainApp::draw()
    {
      gl::enableDepthRead();
      gl::enableDepthWrite();
      gl::clear( Color::black() );
      gl::setViewport(getWindowBounds());
      gl::setMatrices(mMayaCam.getCamera());

      gl::color(Color(1.f,0.f,0.f));
      gl::drawSphere(mRepPosition, 0.25f);

      gl::color(Color::white());
      mParticleSystem.draw();
    }
    ```

1.  将`Particle.cpp`源文件中的`Particle`的`draw`方法替换为以下内容

    ```cpp
    void Particle::draw(){
      glBegin(GL_POINTS);
      glVertex2f(position);
      glEnd();
    }
    ```

## 如何工作…

首先，我们在第 6 步中创建的粒子代替了网格的顶点。

![如何工作…](img/8703OS_5_9.jpg)

你可以在第 8 步找到一个重要的计算，我们尝试将粒子移动回存储在`anchor`属性中的原始位置。为了使粒子偏移，我们使用了与*应用排斥和吸引力的力*菜谱中相同的排斥代码，但对其进行了三维空间的轻微修改。基本上，它涉及到使用`Vec3f`类型而不是`Vec2f`。

![如何工作…](img/8703OS_5_10.jpg)

# 创建弹簧

在这个菜谱中，我们将学习如何创建弹簧。

**弹簧**是连接两个粒子并使它们保持在定义的静止距离的对象。

在这个例子中，我们将创建随机粒子，并且每当用户按下鼠标按钮时，两个随机粒子将通过一个新的弹簧连接，弹簧的静止距离是随机的。

## 准备工作

我们将使用之前菜谱中开发的相同粒子系统，即*在 2D 中创建粒子系统*。创建该菜谱中描述的`Particle`和`ParticleSystem`类，并在应用程序源文件顶部包含`ParticleSystem.h`文件。

我们将创建一个`Spring`类，因此有必要创建以下文件：

+   `Spring.h`

+   `Spring.cpp`

## 如何实现它...

我们将创建约束粒子运动的弹簧。执行以下步骤以实现此目的：

1.  在`Spring.h`文件中，我们将声明一个`Spring`类。首先，我们需要添加`#pragma once`宏并包含必要的文件。

    ```cpp
    #pragma once
    #include "Particle.h"
    #include "cinder/gl/gl.h"
    ```

1.  接下来，声明`Spring`类。

    ```cpp
    class Spring{

    };
    ```

1.  我们将添加成员变量，两个`Particle`指针以引用将通过此弹簧连接的粒子，以及`rest`和`strengthfloat`变量。

    ```cpp
    class Spring{
    public:
      Particle *particleA;
      Particle *particleB;
      float strength, rest;
    };
    ```

1.  现在我们将声明一个构造函数，它将接受两个`Particle`对象的指针以及`rest`和`strength`值。

    我们还将声明`update`和`draw`方法。

    以下为最终的`Spring`类声明：

    ```cpp
    class Spring{
    public:

        Spring( Particle *particleA, Particle *particleB, 
        float rest, float strength );

        void update();
        void draw();

        Particle *particleA;
        Particle *particleB;
        float strength, rest;

    };
    ```

1.  让我们在`Spring.cpp`文件中实现`Spring`类。

    在构造函数中，我们将成员变量的值设置为通过参数传入的值。

    ```cpp
    Spring::Spring( Particle *particleA, Particle *particleB, float rest, float strength ){
      this->particleA = particleA;
      this->particleB = particleB;
      this->rest = rest;
      this->strength = strength;
    }
    ```

1.  在`Spring`类的`update`方法中，我们将计算粒子之间的距离与弹簧的平衡距离之间的差异，并相应地调整它们。

    ```cpp
    void Spring::update(){
        ci::Vec2f delta = particleA->position - particleB->position;
        float length = delta.length();
        float invMassA = 1.0f / particleA->mass;
        float invMassB = 1.0f / particleB->mass;
        float normDist = ( length - rest ) / ( length * ( invMassA + invMassB ) ) * strength;
        particleA->position -= delta * normDist * invMassA;
        particleB->position += delta * normDist * invMassB;
    }
    ```

1.  在`Spring`类的`draw`方法中，我们将简单地绘制一条连接两个粒子的线。

    ```cpp
    void Spring::draw(){
        ci::gl::drawLine
        ( particleA->position, particleB->position );
    }
    ```

1.  现在我们必须在`ParticleSystem`类中进行一些更改，以允许添加弹簧。

    在`ParticleSystem`文件中，包含`Spring.h`文件。

    ```cpp
    #include "Spring.h"
    ```

1.  在类声明中声明`std::vector<Spring*>`成员。

    ```cpp
    std::vector<Spring*> springs;
    ```

1.  声明`addSpring`和`destroySpring`方法，用于向系统中添加和销毁弹簧。

    以下为最终的`ParticleSystem`类声明：

    ```cpp
    classParticleSystem{
    public:

        ~ParticleSystem();

        void update();
        void draw();

        void addParticle( Particle *particle );
        void destroyParticle( Particle *particle );
        void addSpring( Spring *spring );
        void destroySpring( Spring *spring );

        std::vector<Particle*> particles;
        std::vector<Spring*> springs;

    };
    ```

1.  让我们实现`addSpring`方法。在`ParticleSystem.cpp`文件中，添加以下代码片段：

    ```cpp
    void ParticleSystem::addSpring( Spring *spring ){
      springs.push_back( spring );
    }
    ```

1.  在`destroySpring`方法的实现中，我们将找到对应于参数`Spring`的迭代器，并将其从弹簧中移除。我们还将删除该对象。

    在`ParticleSystem.cpp`文件中添加以下代码片段：

    ```cpp
    void ParticleSystem::destroySpring( Spring *spring ){
      std::vector<Spring*>::iterator it = std::find( springs.begin(), springs.end(), spring );
      delete *it;
      springs.erase( it );
    }
    ```

1.  必须修改`update`方法以更新所有弹簧。

    以下代码片段显示了最终的更新应该看起来像什么：

    ```cpp
    void ParticleSystem::update(){
      for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ){
            (*it)->update();
        }
        for( std::vector<Spring*>::iterator it = 
        springs.begin(); it != springs.end(); ++it ){
            (*it)->update();
        }
    }
    ```

1.  在`draw`方法中，我们还需要遍历所有弹簧并调用它们的`draw`方法。

    `ParticleSystem::draw`方法的最终实现应该如下所示：

    ```cpp
    void ParticleSystem::draw(){
        for( std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it ){
            (*it)->draw();
        }
        for( std::vector<Spring*>::iterator it = 
        springs.begin(); it != springs.end(); ++it ){
            (*it)->draw();
        }
    }
    ```

1.  我们已经完成了`Spring`类的创建和对`ParticleSystem`类所有必要更改的制作。

    让我们转到我们的应用程序类并包含`ParticleSystem.h`文件：

    ```cpp
    #include "ParticleSystem.h"
    ```

1.  声明一个`ParticleSystem`对象。

    ```cpp
    ParticleSystem mParticleSystem;
    ```

1.  通过将以下代码片段添加到`setup`方法中，创建一些随机粒子：

    ```cpp
    for( int i=0; i<100; i++ ){
            float x = randFloat( getWindowWidth() );
            float y = randFloat( getWindowHeight() );
            float radius = randFloat( 5.0f, 15.0f );
            float mass = radius;
            float drag = 0.9f;
            Particle *particle = 
            new Particle( Vec2f( x, y ), radius, mass, drag );
            mParticleSystem.addParticle( particle );
        }
    ```

1.  在`update`方法中，我们需要调用`ParticleSystem`上的`update`方法。

    ```cpp
    void MyApp::update(){
      mParticleSystem.update();
    }
    ```

1.  在`draw`方法中，清除背景，定义窗口的矩阵，并在`mParticleSystem`上调用`draw`方法。

    ```cpp
    void MyApp::draw(){
      gl::clear( Color( 0, 0, 0 ) );
      gl::setMatricesWindow( getWindowWidth(), getWindowHeight() );
      mParticleSystem.draw();
    }
    ```

1.  由于我们希望在用户按下鼠标时创建弹簧，因此我们需要声明`mouseDown`方法。

    将以下代码片段添加到应用程序的类声明中：

    ```cpp
      void mouseDown( MouseEvent event );
    ```

1.  在`mouseDown`实现中，我们将连接两个随机粒子。

    首先声明一个`Particle`指针并将其定义为粒子系统中的一个随机粒子。

    ```cpp
    Particle *particleA = mParticleSystem.particles[ randInt( mParticleSystem.particles.size() ) ];
    ```

1.  现在声明第二个`Particle`指针并将其设置为等于第一个指针。在`while`循环中，我们将将其值设置为`mParticleSystem`中的随机粒子，直到两个粒子不同。这将避免两个指针都指向同一粒子的情形。

    ```cpp
    Particle *particleB = particleA;
    while( particleB == particleA ){
      particleB = mParticleSystem.particles[ randInt( mParticleSystem.particles.size() ) ];
        }
    ```

1.  现在我们将创建一个`Spring`对象，它将连接两个粒子，定义一个随机的静止距离，并将`strength`设置为`1.0`。将创建的弹簧添加到`mParticleSystem`中。

    以下是最终的`mouseDown`实现：

    ```cpp
    void SpringsApp::mouseDown( MouseEvent event )
    {
        Particle *particleA = mParticleSystem.particles[ 
        randInt( mParticleSystem.particles.size() ) ];
        Particle *particleB = particleA;
        while( particleB == particleA ){
      particleB = mParticleSystem.particles[ randInt( mParticleSystem.particles.size() ) ];
        }
        float rest = randFloat( 100.0f, 200.0f );
        float strength = 1.0f;
        Spring *spring = new Spring
        ( particleA, particleB, rest, strength );
        mParticleSystem.addSpring( spring );

    }
    ```

1.  构建并运行应用程序。每次按下鼠标按钮时，两个粒子将通过一条白色线条连接，并且它们的距离将保持不变。![如何操作…](img/87030s_5_11.jpg)

## 它是如何工作的...

`Spring`对象将计算两个粒子之间的差异并修正它们的位置，以便两个粒子之间的距离等于弹簧的静止值。

通过使用它们的质量，我们还将考虑每个粒子的质量，因此修正将考虑粒子的重量。

## 还有更多...

同样的原理也可以应用于 3D 粒子系统。

如果你正在使用 3D 粒子，如*在 2D 中创建粒子系统*食谱的*更多内容…*部分所述，`Spring`类只需将其计算更改为使用`ci::Vec3f`而不是`ci::Vec2f`。

`Spring`类的`update`方法将类似于以下代码片段：

```cpp
void Spring::update(){
    ci::Vec3f delta = particleA->position - particleB->position;
    float length = delta.length();
    float invMassA = 1.0f / particleA->mass;
    float invMassB = 1.0f / particleB->mass;
    float normDist = ( length - rest ) / ( length * ( invMassA + invMassB ) ) * strength;
    particleA->position -= delta * normDist * invMassA;
    particleB->position += delta * normDist * invMassB;
}
```
