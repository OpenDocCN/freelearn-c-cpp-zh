# OpenGL and Vulkan in Qt applications

Hardware acceleration is crucial for implementing modern games with advanced graphics effects. Qt Widgets module uses traditional approach optimized for CPU-based rendering. Even though you can make any widget use OpenGL, the performance will usually not be maximized. However, Qt allows you to use OpenGL or Vulkan directly to create high-performance graphics limited only by the graphics card's processing power. In this chapter, you will learn about employing your OpenGL and Vulkan skills to display fast 3D graphics. If you are not familiar with these technologies, this chapter should give you a kickstart for further research in this topic. We will also describe multiple Qt helper classes that simplify usage of OpenGL textures, shaders, and buffers. By the end of the chapter, you will be able to create 2D and 3D graphics for your games using OpenGL and Vulkan classes offered by Qt and integrate them with the rest of the user interface.

The main topics covered in this chapter are as listed:

*   OpenGL in Qt applications
*   Immediate mode
*   Textures
*   Shaders
*   OpenGL buffers
*   Vulkan in Qt applications

# Introduction to OpenGL with Qt

We are not experts on OpenGL, so in this part of the chapter, we will not teach you to do any fancy stuff with OpenGL and Qt but will show you how to enable the use of your OpenGL skills in Qt applications. There are a lot of tutorials and courses on OpenGL out there, so if you're not that skilled with OpenGL, you can still benefit from what is described here by employing the knowledge gained here to more easily learn fancy stuff. You can use external materials and a high-level API offered by Qt, which will speed up many of the tasks described in the tutorials.

# OpenGL windows and contexts

There are many ways you can perform OpenGL rendering in Qt. The most straightforward way that we will mainly use is to subclass `QOpenGLWindow`. It allows OpenGL to render your content directly to a whole window and is suitable if you draw everything in your application with OpenGL. You can make it a fullscreen window if you want. However, later we will also discuss other approaches that will allow you to integrate OpenGL content into a widget-based application.

The OpenGL context represents the overall state of the OpenGL pipeline, which guides the process of data processing and rendering to a particular device. In Qt, it is represented by the `QOpenGLContext` class. A related concept that needs explanation is the idea of an OpenGL context being "current" in a thread. The way OpenGL calls work is that they do not use any handle to any object containing information on where and how to execute the series of low-level OpenGL calls. Instead, it is assumed that they are executed in the context of the current machine state. The state may dictate whether to render a scene to a screen or to a frame buffer object, which mechanisms are enabled, or the properties of the surface OpenGL is rendering on. Making a context "current" means that all further OpenGL operations issued by a particular thread will be applied to this context. To add to that, a context can be "current" only in one thread at the same time; therefore, it is important to make the context current before making any OpenGL calls and then marking it as available after you are done accessing OpenGL resources.

`QOpenGLWindow` has a very simple API that hides most of the unnecessary details from the developer. Apart from constructors and a destructor, it provides a small number of very useful methods. First, there are auxiliary methods for managing the OpenGL context: `context()`, which returns the context, and `makeCurrent()` as well as `doneCurrent()` for acquiring and releasing the context. The class also provides a number of virtual methods we can re-implement to display OpenGL graphics.

We will be using the following three virtual methods:

*   `initializeGL()` is invoked by the framework once, before any painting is actually done so that you can prepare any resources or initialize the context in any way you require.
*   `paintGL()` is the equivalent of `paintEvent()` for the widget classes. It gets executed whenever the window needs to be repainted. This is the function where you should put your OpenGL rendering code.
*   `resizeGL()` is invoked every time the window is resized. It accepts the width and height of the window as parameters. You can make use of that method by re-implementing it so that you can prepare yourself for the fact that the next call to `paintGL()` renders to a viewport of a different size.

Before calling any of these virtual functions, `QOpenGLWindow` ensures that the OpenGL context is current, so there is no need to manually call `makeCurrent()` in them.

# Accessing OpenGL functions

Interaction with OpenGL is usually done through calling functions provided by the OpenGL library. For example, in a regular C++ OpenGL application, you can see calls to OpenGL functions such as `glClearColor()`. These functions are resolved when your binary is linked against the OpenGL library. However, when you write a cross-platform application, resolving all the required OpenGL functions is not trivial. Luckily, Qt provides a way to call OpenGL functions without having to worry about the platform-specific details.

In a Qt application, you should access OpenGL functions through a family of `QOpenGLFunctions` classes. The `QOpenGLFunctions` class itself only provides access to functions that are part of OpenGL ES 2.0 API. This subset is expected to work at most desktop and embedded platforms supported by Qt (where OpenGL is available at all). However, this is a really limited set of functions, and sometimes you may want to use a more recent OpenGL version at the cost of supporting less platforms. For each known OpenGL version and profile, Qt provides a separate class that contains the set of available functions. For example, the `QOpenGLFunctions_3_3_Core` class will contain all functions provided by the OpenGL 3.3 core profile.

The approach recommended by Qt is to select the OpenGL functions class corresponding to the version you want to use and add this class an the second base class of your window or widget. This will make all OpenGL functions from that version available within your class. This approach allows you to use code that was using the OpenGL library directly without changing it. When you put such code in your class, the compiler will, for example, use the `QOpenGLFunctions::glClearColor()` function instead of the global `glClearColor()` function provided by the OpenGL library.

However, when using this approach, you must be careful to only use functions provided by your base class. You can accidentally use a global function instead of a function provided by Qt classes if the Qt class you choose does not contain it. For example, if you use `QOpenGLFunctions` as the base class, you can't use the `glBegin()` function, as it is not provided by this Qt class. Such erroneous code may work on one operating system and then suddenly not compile on another because you don't link against the OpenGL library. As long as you only use OpenGL functions provided by Qt classes, you don't have to think about linking with the OpenGL library or resolving functions in a cross-platform way.

If you want to ensure that you only use Qt OpenGL function wrappers, you can use the Qt class as a private field instead of a base class. In that case, you have to access every OpenGL function through the private field, for example, `m_openGLFunctions->glClearColor()`. This will make your code more verbose, but at least you will be sure that you don't accidentally use a global function.

Before using Qt OpenGL functions, you have to call the `initializeOpenGLFunctions()` method of the functions class in the current OpenGL context. This is usually done in the `initializeGL()` function of the window. The `QOpenGLFunctions` class is expected to always initialize successfully, so its `initializeOpenGLFunctions()` method doesn't return anything. In all the other functions' classes, this function returns `bool`. If it returns `false`, it means that Qt was not able to resolve all the required functions successfully, and your application should exit with an error message.

In our examples, we will use the `QOpenGLFunctions_1_1` class that contains all OpenGL functions we'll use. When you're creating your own project, think about the OpenGL profile you want to target and select the appropriate functions class.

# Using OpenGL in immediate mode

We will start with the most basic approach that's called **immediate mode**. In this mode, no additional setup of OpenGL buffers or shaders is required. You can just supply a bunch of geometric primitives and get the result right away. Immediate mode is now deprecated because it works much slower and is less flexible than more advanced techniques. However, it's so much easier than them that basically every OpenGL tutorial starts with describing the immediate mode calls. In this section, we'll show how to perform some simple OpenGL drawing with very little code. A more modern approach will be covered in the next section of this chapter.

# Time for action – Drawing a triangle using Qt and OpenGL

For the first exercise, we will create a subclass of `QOpenGLWindow` that renders a triangle using simple OpenGL calls. Create a new project, starting with Empty qmake Project from the Other Project group as the template. In the project file, put the following content:

```cpp
QT = core gui
TARGET = triangle
TEMPLATE = app 
```

Note that our project does not include Qt Widgets module. Using the `QOpenGLWindow` approach allows us to remove this unnecessary dependency and make our application more lightweight.

Note that Qt Core and Qt GUI modules are enabled by default, so you don't have to add them to the `QT` variable, but we prefer to explicitly show that we are using them in our project.

Having the basic project setup ready, let's define a `SimpleGLWindow` class as a subclass of `QOpenGLWindow` and `QOpenGLFunctions_1_1`. Since we don't want to allow external access to OpenGL functions, we use protected inheritance for the second base. Next, we override the virtual `initializeGL()` method of `QOpenGLWindow`. In this method, we initialize our `QOpenGLFunctions_1_1` base class and use the `glClearColor()` function that it provides:

```cpp
class SimpleGLWindow : public QOpenGLWindow,
                       protected QOpenGLFunctions_1_1 
{
public:
    SimpleGLWindow(QWindow *parent = 0) :
        QOpenGLWindow(NoPartialUpdate, parent) {
    }
protected:
    void initializeGL() {
        if (!initializeOpenGLFunctions()) {
            qFatal("initializeOpenGLFunctions failed");
        }
        glClearColor(1, 1, 1, 0);
    } 
};
```

In `initializeGL()`, we first call `initializeOpenGLFunctions()`, which is a method of the `QOpenGLFunctions_1_1` class, one of the base classes of our window class. The method takes care of setting up all the functions according to the parameters of the current OpenGL context (thus, it is important to first make the context current, which luckily is done for us behind the scenes before `initializeGL()` is invoked). If this function fails, we use the `qFatal()` macro to print an error message to `stderr` and abort the application. Then, we use the `QOpenGLFunctions_1_1::glClearColor()` function to set the clear color of the scene to white.

The next step is to re-implement `paintGL()` and put the actual drawing code there:

```cpp
void SimpleGLWindow::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width(), height());
    glBegin(GL_TRIANGLES);
    {
        glColor3f(1, 0, 0);
        glVertex3f( 0.0f, 1.0f, 0.0f);
        glColor3f(0, 1, 0);
        glVertex3f( 1.0f, -1.0f, 0.0f);
        glColor3f(0, 0, 1);
        glVertex3f(-1.0f, -1.0f, 0.0f);
    }
    glEnd();
}
```

This function first clears the color buffer and sets the OpenGL viewport of the context to be the size of the window. Then, we tell OpenGL to start drawing using triangles with the `glBegin()` call and passing `GL_TRIANGLES` as the drawing mode. Then, we pass three vertices along with their colors to form a triangle. Finally, we inform the pipeline, by invoking `glEnd()`, that we are done drawing using the current mode.

What is left is a trivial `main()` function that sets up the window and starts the event loop. Add a new C++ Source File, call it `main.cpp`, and implement `main()`, as follows:

```cpp
int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    SimpleGLWindow window;
    window.resize(600, 400);
    window.show();
    return app.exec();
} 
```

This function is very similar to what we usually have in the `main()` function, but we use `QGuiApplication` instead of `QApplication`, because we only use the Qt GUI module. After running the project, you should see the following:

![](img/70d987b1-b8d0-4f23-ad39-331b02b74d7c.png)

# Multisampling

You can see that the triangle has jagged edges. That's because of the aliasing effect. You can counter it by enabling multisampling for the window, which will make OpenGL render the contents as if the screen had higher resolution and then average the result, which acts as anti-aliasing. To do that, add the following code to the constructor of the window:

```cpp
QSurfaceFormat fmt = format();
fmt.setSamples(16); // multisampling set to 16
setFormat(fmt);
```

Note that multisampling is resource-demanding, so setting a high number of samples may cause your application to fail if your hardware or driver can't handle it. If the application doesn't work after enabling multisampling, try to lower the number of samples or just disable it.

# Time for action – Scene-based rendering

Let's take our rendering code to a higher level. Putting OpenGL code directly into the window class requires subclassing the window class and makes the window class more and more complex. Let's follow good programming practice and separate rendering code from window code.

Create a new class and call it `AbstractGLScene`. It will be the base class for
definitions of OpenGL scenes. We also derive the class (with protected scope) from `QOpenGLFunctions_1_1` to make accessing different OpenGL functions easier. Make the scene class accept a pointer to `QOpenGLWindow`, either in the constructor or through a dedicated setter method. Ensure that the pointer is stored in the class for easier access, as we will rely on that pointer for accessing physical properties of the window. Add methods for querying the window's OpenGL context. You should end up with code similar to the following:

```cpp
class AbstractGLScene : protected QOpenGLFunctions_1_1 {
public:
    AbstractGLScene(QOpenGLWindow *window = nullptr) {
        m_window = window;
    }
    QOpenGLWindow* window() const { return m_window; }
    QOpenGLContext* context() {
       return m_window ? m_window->context() : nullptr;
    }
    const QOpenGLContext* context() const {
        return m_window ? m_window->context() : nullptr;
    }
private:
    QOpenGLWindow *m_window = nullptr;
}; 
```

Now the essential part begins. Add two pure virtual methods called `paint()` and `initialize()`. Also, remember to add a virtual destructor.

Instead of making `initialize()` a pure virtual function, you can implement its body in such a way that it will call `initializeOpenGLFunctions()` to fulfill the requirements of the `QOpenGFunctions` class. Then, subclasses of `AbstractGLScene` can ensure that the functions are initialized properly by calling the base class implementation of `initialize()`.

Next, create a subclass of `QOpenGLWindow` and call it `SceneGLWindow`. Add an `AbstractGLScene *m_scene` private field and implement a getter and a setter for it. Create a constructor using the following code:

```cpp
SceneGLWindow::SceneGLWindow(QWindow *parent) : 
    QOpenGLWindow(NoPartialUpdate, parent)
{
}
```

This constructor forwards the parent argument to the base constructor and assigns `NoPartialUpdate` as the window's `UpdateBehavior`. This option means that the window will be fully painted on each `paintGL()` call and thus no framebuffer is needed. This is the default value of the first argument, but since we provide the second argument, we are obligated to provide the first argument explicitly.

Then, re-implement the `initializeGL()` and `paintGL()` methods and make them call appropriate equivalents in the scene:

```cpp
void SceneGLWindow::initializeGL() {
    if(m_scene) {
        m_scene->initialize();
    }
}
void SceneGLWindow::paintGL() {
    if(m_scene) {
        m_scene->paint();
    }
}
```

Finally, instantiate `SceneGLWindow` in the `main()` function.

# What just happened?

We have just set up a class chain that separates the window code from the actual OpenGL scene. The window forwards all calls related to scene contents to the scene object so that when the window is requested to repaint itself, it delegates the task to the scene object. Note that prior to doing that, the window will make the OpenGL context current; therefore, all OpenGL calls that the scene makes will be related to that context. You can store the code created in this exercise for later reuse in further exercises and your own projects.

# Time for action – Drawing a textured cube

Create a new class named `CubeGLScene` and derive it from `AbstractGLScene`. Implement the constructor to forward its argument to the base class constructor. Add a method to store a `QImage` object in the scene that will contain texture data for the cube. Add a `QOpenGLTexture` pointer member as well, which will contain the texture, initialize it to `nullptr` in the constructor, and delete it in the destructor. Let's call the `m_textureImage` image object and the `m_texture` texture. Now add a protected `initializeTexture()` method and fill it with the following code:

```cpp
void CubeGLScene::initializeTexture() {
    m_texture = new QOpenGLTexture(m_textureImage.mirrored());
    m_texture->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    m_texture->setMagnificationFilter(QOpenGLTexture::Linear);
} 
```

The function first mirrors the image vertically. This is because the *y* axis in OpenGL points up by default, so a texture would be displayed "upside down". Then, we create a `QOpenGLTexture` object, passing it our image. After that, we set minification and magnification filters so that the texture looks better when it is scaled.

We are now ready to implement the `initialize()` method that will take care of setting up the texture and the scene itself:

```cpp
void CubeGLScene::initialize() {
    AbstractGLScene::initialize();
    m_initialized = true;
    if(!m_textureImage.isNull()) {
        initializeTexture();
    }
    glClearColor(1, 1, 1, 0);
    glShadeModel(GL_SMOOTH);
} 
```

We make use of a flag called `m_initialized`. This flag is needed to prevent the texture from being set up too early (when no OpenGL context is available yet). Then, we check whether the texture image is set (using the `QImage::isNull()` method); if so, we initialize the texture. Then, we set some additional properties of the OpenGL context.

In the setter for `m_textureImage`, add code that checks whether `m_initialized` is set to `true` and, if so, calls `initializeTexture()`. This is to make certain that the texture is properly set regardless of the order in which the setter and `initialize()` are called. Also remember to set `m_initialized` to `false` in the constructor.

The next step is to prepare the cube data. We will define a special data structure for the cube that groups vertex coordinates and texture data in a single object. To store coordinates, we will use classes tailored to that purpose—`QVector3D` and `QVector2D`:

```cpp
struct TexturedPoint {
    QVector3D coord;
    QVector2D uv;
    TexturedPoint(const QVector3D& pcoord = QVector3D(),
                  const QVector2D& puv = QVector2D()) :
        coord(pcoord), uv(puv) {
    }
}; 
```

`QVector2D`, `QVector3D`, and `QVector4D` are helper classes that represent a single point in space and provide some convenient methods. For instance, `QVector2D` stores two `float` variables (`x` and `y`), much like the `QPointF` class does. These classes are not to be confused with `QVector<T>`, a container template class that stores a collection of elements.

`QVector<TexturedPoint>` will hold information for the whole cube. The vector is initialized with data using the following code:

```cpp
void CubeGLScene::initializeCubeData() {
    m_data = {
        // FRONT FACE
        {{-0.5, -0.5,  0.5}, {0, 0}}, {{ 0.5, -0.5,  0.5}, {1, 0}},
        {{ 0.5,  0.5,  0.5}, {1, 1}}, {{-0.5,  0.5,  0.5}, {0, 1}},

        // TOP FACE
        {{-0.5,  0.5,  0.5}, {0, 0}}, {{ 0.5,  0.5,  0.5}, {1, 0}},
        {{ 0.5,  0.5, -0.5}, {1, 1}}, {{-0.5,  0.5, -0.5}, {0, 1}},
        //...
    };
} 
```

The code uses C++11 initializer list syntax to set the vector's data. The cube consists of six faces and is centered on the origin of the coordinate system. The following diagram presents the same data in graphical form:

![](img/3957ca2b-bdee-41e6-99ff-b5c2d0b77fad.png)

`initializeCubeData()` should be called from the scene constructor or from the `initialize()` method. What remains is the painting code:

```cpp
  void CubeGLScene::paint() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, window()->width(), window()->height());
    glLoadIdentity();

    glRotatef(45, 1.0, 0.0, 0.0);
    glRotatef(45, 0.0, 1.0, 0.0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    paintCube();
} 
```

First, we set up the viewport and then we rotate the view. Before calling `paintCube()`, which will render the cube itself, we enable depth testing and face culling so that only visible faces are drawn. The `paintCube()` routine looks as follows:

```cpp
void CubeGLScene::paintCube() {
    if(m_texture) {
        m_texture->bind();
    }
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    for(const TexturedPoint &point: m_data) {
        glTexCoord2d(point.uv.x(), point.uv.y());
        glVertex3f(point.coord.x(), point.coord.y(), point.coord.z());
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
} 
```

First, the texture is bound and texturing is enabled. Then, we enter the quad drawing mode and stream in data from our data structure. Finally, we disable texturing again.

For completeness, here's a `main()` function that executes the scene:

```cpp
int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    SceneGLWindow window;
    QSurfaceFormat fmt;
    fmt.setSamples(16);
    window.setFormat(fmt);
    CubeGLScene scene(&window);
    window.setScene(&scene);
    scene.setTexture(QImage(":/texture.jpg"));
    window.resize(600, 600);
    window.show();
    return app.exec();
} 
```

Note the use of `QSurfaceFormat` to enable multisample antialiasing for the scene. We have also put the texture image into a resource file to avoid problems with the relative path to the file.

# Have a go hero – Animating a cube

Try modifying the code to make the cube animated. To do that, have the scene inherit `QObject`, add an angle property of the `float` type to it (remember about the `Q_OBJECT` macro). Then, modify one of the `glRotatef()` lines to use the angle value instead of a constant value. Put the following code in `main()`, right before calling `app.exec()`:

```cpp
QPropertyAnimation animation(&scene, "angle");
animation.setStartValue(0);
animation.setEndValue(359);
animation.setDuration(5000);
animation.setLoopCount(-1);
animation.start();
```

Remember to put a call to `window()->update()` in the setter for the angle property so that the scene is redrawn.

# Modern OpenGL with Qt

The OpenGL code shown in the previous section uses a very old technique of streaming vertices one by one into a fixed OpenGL pipeline. Nowadays, modern hardware is much more feature-rich and not only does it allow faster processing of vertex data but also offers the ability to adjust different processing stages, with the use of reprogrammable units called **shaders**. In this section, we will take a look at what Qt has to offer in the domain of a "modern" approach to using OpenGL.

# Shaders

Qt can make use of shaders through a set of classes based around `QOpenGLShaderProgram`. This class allows compiling, linking, and executing of shader programs written in GLSL. You can check whether your OpenGL implementation supports shaders by inspecting the result of a static `QOpenGLShaderProgram::hasOpenGLShaderPrograms()` call that accepts a pointer to an OpenGL context. All modern hardware and all decent graphics drivers should have some support for shaders.

Qt supports all kinds of shaders, with the most common being vertex and fragment shaders. These are both part of the classic OpenGL pipeline. You can see an illustration of the pipeline in the following diagram:

![](img/f42b8281-ae7a-41a0-b990-31cdd60f284b.png)

A single shader is represented by an instance of the `QOpenGLShader` class. You need to specify the type of the shader in the constructor of this class. Then, you can compile the shader's source code by calling `QOpenGLShader::compileSourceCode()`, which has a number of overloads for handling different input formats, or `QOpenGLShader::compileSourceFile()`. The `QOpenGLShader` object stores the ID of the compiled shader for future use.

When you have a set of shaders defined, you can assemble a complete program using `QOpenGLShaderProgram::addShader()`. After all shaders are added, you can `link()` the program and `bind()` it to the current OpenGL context. The program class has a number of methods for setting values of different input parameters—uniforms and attributes both in singular and array versions. Qt provides mappings between its own types (such as `QSize` or `QColor`) to GLSL counterparts (for example, `vec2` and `vec4`) to make the programmer's life even easier.

A typical code flow for using shaders for rendering is as follows (first a vertex shader is created and compiled):

```cpp
QOpenGLShader vertexShader(QOpenGLShader::Vertex);
vertexShader.compileSourceCode(
    "uniform vec4 color;\n"
    "uniform highp mat4 matrix;\n"
    "void main(void) { gl_Position = gl_Vertex * matrix; }"
);
```

The process is repeated for a fragment shader:

```cpp
QOpenGLShader fragmentShader(QOpenGLShader::Fragment);
fragmentShader.compileSourceCode(
    "uniform vec4 color;\n"
    "void main(void) { gl_FragColor = color; }"
);
```

Then, shaders are linked into a single program in a given OpenGL context:

```cpp
QOpenGLShaderProgram program(context);
program.addShader(&vertexShader);
program.addShader(&fragmentShader);
program.link(); 
```

When shaders are linked together, OpenGL searches for common variables (such as uniforms or buffers) in them and maps them together. This allows you, for example, to pass a value from the vertex shader to the fragment shader. Behind the scenes, the `link()` function uses the `glLinkProgram()` OpenGL call.

Whenever the program is used, it should be bound to the current OpenGL context and filled with the required data:

```cpp
program.bind();
QMatrix4x4 matrix = /* ... */;
QColor color = Qt::red;
program.setUniformValue("matrix", matrix);
program.setUniformValue("color", color); 
```

After that, calls activating the render pipeline will use the bound program:

```cpp
glBegin(GL_TRIANGLE_STRIP);
//...
glEnd(); 
```

# Time for action – Shaded objects

Let's convert our last program so that it uses shaders. To make the cube better, we will implement a smooth lighting model using the Phong algorithm. At the same time, we will learn to use some helper classes that Qt offers for use with OpenGL.

The basic goals for this miniproject are as follows:

*   Use vertex and fragment shaders for rendering a complex object
*   Handle model, view, and projection matrices
*   Use attribute arrays for faster drawing

Start by creating a new subclass of `AbstractGLScene`. Let's give it the following interface:

```cpp
class ShaderGLScene : public QObject, public AbstractGLScene {
    Q_OBJECT
public:
    ShaderGLScene(SceneGLWindow *window);
    void initialize();
    void paint();
protected:
    void initializeObjectData();
private:
    struct ScenePoint {
        QVector3D coords;
        QVector3D normal;
        ScenePoint(const QVector3D &c = QVector3D(),
                   const QVector3D &n = QVector3D()) :
            coords(c), normal(n)
        {
        }
    };
    QOpenGLShaderProgram m_shader;
    QMatrix4x4 m_modelMatrix;
    QMatrix4x4 m_viewMatrix;
    QMatrix4x4 m_projectionMatrix;
    QVector<ScenePoint> m_data;
}; 
```

We're not using textures in this project, so `TexturedPoint` was simplified to `ScenePoint` with UV texture coordinates removed. Update the `main()` function to use the `ShaderGLScene` class.

We can start implementing the interface with the `initializeObjectData()` function that will be called in the constructor. This function must fill the `m_data` member with information about vertices and their normals. The implementation will depend on the source of your data.

In the sample code that comes with this book, you can find code that loads data from a file in the PLY format generated with the Blender 3D program. To export a model from Blender, ensure that it consists of just triangles (for that, select the model, go into the Edit mode by pressing *Tab*, open the Faces menu with *Ctrl* + *F*, and choose Triangulate Faces). Then, click on File and Export; choose Stanford (.ply). You will end up with a text file containing vertex and normal data as well as face definitions for the vertices. We add the PLY file to the project's resources so that it is always available to our program. Then, we use the `PlyReader` C++ class that implements the parsing.

You can always reuse the cube object from the previous project. Just be aware that its normals are not calculated properly for smooth shading; thus, you will have to correct them.

Before we can set up the shader program, we have to be aware of what the actual shaders look like. Shader code will be loaded from external files, so the first step is to add a new file to the project. In Creator, right-click on the project in the project tree and choose Add New...; from the left pane, choose GLSL, and from the list of available templates, choose Vertex Shader (Desktop OpenGL). Call the new file `phong.vert` and input the following code:

```cpp
uniform highp mat4 modelViewMatrix;
uniform highp mat3 normalMatrix;
uniform highp mat4 projectionMatrix;
uniform highp mat4 mvpMatrix;

attribute highp vec4 Vertex;
attribute mediump vec3 Normal;

varying mediump vec3 N;
varying highp vec3 v;

void main(void) {
    N = normalize(normalMatrix * Normal);
    v = vec3(modelViewMatrix * Vertex);
    gl_Position = mvpMatrix * Vertex;
} 
```

The code is very simple. We declare four matrices representing different stages of coordinate mapping for the scene. We also define two input attributes—`Vertex` and `Normal`—which contain the vertex data. The shader will output two pieces of data—a normalized vertex normal and a transformed vertex coordinate as seen by the camera. Of course, apart from that, we set `gl_Position` to be the final vertex coordinate. In each case, we want to be compliant with the OpenGL/ES specification, so we prefix each variable declaration with a precision specifier.

Next, add another file, call it `phong.frag`, and make it a fragment shader (Desktop OpenGL). The content of the file is a typical ambient, diffuse, and specular calculation:

```cpp
struct Material {
    lowp vec3 ka;
    lowp vec3 kd;
    lowp vec3 ks;
    lowp float shininess;
};

struct Light {
    lowp vec4 position;
    lowp vec3 intensity;
};

uniform Material mat;

uniform Light light;

varying mediump vec3 N;
varying highp vec3 v;

void main(void) {
    vec3 n = normalize(N);
    vec3 L = normalize(light.position.xyz - v);
    vec3 E = normalize(-v);
    vec3 R = normalize(reflect(-L, n));

    float LdotN = dot(L, n);
    float diffuse = max(LdotN, 0.0);
    vec3 spec = vec3(0, 0, 0);

    if(LdotN > 0.0) {
        float RdotE = max(dot(R, E), 0.0);
        spec = light.intensity * pow(RdotE, mat.shininess);
    }
    vec3 color = light.intensity * (mat.ka + mat.kd * diffuse + mat.ks * spec);
    gl_FragColor = vec4(color, 1.0);
}
```

Apart from using the two varying variables to obtain the interpolated normal (`N`) and fragment (`v`) position, the shader declares two structures for keeping light and material information. Without going into the details of how the shader itself works, it calculates three components—ambient light, diffused light, and specular reflection—adds them together, and sets that as the fragment color. Since all the per vertex input data is interpolated for each fragment, the final color is calculated individually for each pixel.

Once we know what the shaders expect, we can set up the shader program object. Let's go through the `initialize()` method. First, we call the base class implementation and set the background color of the scene to black, as shown in the following code:

```cpp
void initialize() {
    AbstractGLScene::initialize();
    glClearColor(0, 0, 0, 0);
    //...
}
```

Add both shader files to the project's resources. Then, use the following code to read shaders from these files and link the shader program:

```cpp
m_shader.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/phong.vert");
m_shader.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/phong.frag");
m_shader.link();
```

The `link()` function returns a Boolean value, but, we skip the error check here for simplicity. The next step is to prepare all the input data for the shader, as shown:

```cpp
m_shader.bind();
m_shader.setAttributeArray("Vertex", GL_FLOAT,
                           &m_data[0].coords, 3, sizeof(ScenePoint));
m_shader.enableAttributeArray("Vertex");

m_shader.setAttributeArray("Normal", GL_FLOAT,
                           &m_data[0].normal, 3, sizeof(ScenePoint));
m_shader.enableAttributeArray("Normal");

m_shader.setUniformValue("mat.ka", QVector3D(0.1, 0, 0.0));
m_shader.setUniformValue("mat.kd", QVector3D(0.7, 0.0, 0.0));
m_shader.setUniformValue("mat.ks", QVector3D(1.0, 1.0, 1.0));
m_shader.setUniformValue("mat.shininess", 128.0f);

m_shader.setUniformValue("light.position", QVector3D(2, 1, 1));
m_shader.setUniformValue("light.intensity", QVector3D(1, 1, 1));
```

First, the shader program is bound to the current context so that we can operate on it. Then, we enable the setup of two attribute arrays—one for vertex coordinates and the other for their normals. In our program, the data is stored in a `QVector<ScenePoint>`, where each `ScenePoint` has `coords` and normal `fields`, so there are no separate C++ arrays for coordinates and normals. Fortunately, OpenGL is smart enough to use our memory layout as is. We just need to map our vector to two attribute arrays.

We inform the program that an attribute called `Vertex` is an array. Each item of that array consists of three values of the `GL_FLOAT` type. The first array item is located at `&m_data[0].coords`, and data for the next vertex is located at `sizeof(ScenePoint)` bytes later than the data for the current point. Then we have a similar declaration for the `Normal` attribute, with the only exception that the first piece of data is stored at `&m_data[0].normal`. By informing the program about layout of the data, we allow it to quickly read all the vertex information when needed.

After attribute arrays are set, we pass values for uniform variables to the shader program, which concludes the shader program setup. You will note that we didn't set values for uniforms representing the various matrices; we will do that separately for each repaint. The `paint()` method takes care of that:

```cpp
void ShaderGLScene::paint() {
    m_projectionMatrix.setToIdentity();
    float aspectRatio = qreal(window()->width()) / window()->height();
    m_projectionMatrix.perspective(90, aspectRatio, 0.5, 40);

    m_viewMatrix.setToIdentity();
    QVector3D eye(0, 0, 2);
    QVector3D center(0, 0, 0);
    QVector3D up(0, 1, 0);
    m_viewMatrix.lookAt(eye, center, up);
    //...
}
```

In this method, we make heavy use of the `QMatrix4x4` class that represents a 4 × 4 matrix in a so-called row-major order, which is suited to use with OpenGL. At the beginning, we reset the projection matrix and use the `perspective()` method to give it a perspective transformation based on the current window size. Afterward, the view matrix is also reset and the `lookAt()` method is used to prepare the transformation for the camera; center value indicates the center of the view that the eye is looking at. The `up` vector dictates the vertical orientation of the camera (with respect to the eye position).

The next couple of lines are similar to what we had in the previous project:

```cpp
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
glViewport(0, 0, window()->width(), window()->height());
glEnable(GL_DEPTH_TEST);
glEnable(GL_CULL_FACE);
glCullFace(GL_BACK); 
```

After that, we do the actual painting of the object:

```cpp
m_modelMatrix.setToIdentity();
m_modelMatrix.rotate(45, 0, 1, 0);
QMatrix4x4 modelViewMatrix = m_viewMatrix * m_modelMatrix;
paintObject(modelViewMatrix);
```

We start by setting the model matrix, which dictates where the rendered object is positioned relative to the center of the world (in this case, we say that it is rotated 45 degrees around the *y* axis). Then we assemble the model-view matrix (denoting the position of the object relative to the camera) and pass it to the `paintObject()` method:

```cpp
void ShaderGLScene::paintObject(const QMatrix4x4& mvMatrix) {
    m_shader.bind();
    m_shader.setUniformValue("projectionMatrix", m_projectionMatrix);
    m_shader.setUniformValue("modelViewMatrix", mvMatrix);
    m_shader.setUniformValue("mvpMatrix", m_projectionMatrix*mvMatrix);
    m_shader.setUniformValue("normalMatrix", mvMatrix.normalMatrix());
    glDrawArrays(GL_TRIANGLES, 0, m_data.size());
} 
```

This method is very easy, since most of the work was done when setting up the shader program. First, the shader program is activated, and then all the required matrices are set as uniforms for the shader. Included is the normal matrix calculated from the model-view matrix. Finally, a call to `glDrawArrays()` is issued, telling it to render with the `GL_TRIANGLES` mode using active arrays, starting from the beginning of the array (offset `0`) and reading in the `m_data.size()` entities from the array.

After you run the project, you should get a result similar to the following one, which happens to contain the Blender monkey, Suzanne:

![](img/1ce8c466-2324-4654-acd9-73b0b248fc1c.png)

# GL buffers

Using attribute arrays can speed up programming, but for rendering all data still needs to be copied to the graphics card on each use. This can be avoided with OpenGL buffer objects. Qt provides a neat interface for such objects with its `QOpenGLBuffer` class. The currently supported buffer types are vertex buffers (where the buffer contains vertex information), index buffers (where the content of the buffer is a set of indexes to other buffers that can be used with `glDrawElements()`), and also less-commonly-used pixel pack buffers and pixel unpack buffers. The buffer is essentially a block of memory that can be uploaded to the graphics card and stored there for faster access. There are different usage patterns available that dictate how and when the buffer is transferred between the host memory and the GPU memory. The most common pattern is a one-time upload of vertex information to the GPU that can later be referred to during rendering as many times as needed. Changing an existing application that uses an attribute array to use vertex buffers is very easy. First, a buffer needs to be instantiated:

```cpp
ShaderGLScene::ShaderGLScene(SceneGLWindow *window) :
    AbstractGLScene(window), m_vertexBuffer(QOpenGLBuffer::VertexBuffer)
{ /* ... */ }
```

Then, its usage pattern needs to be set. In case of a one-time upload, the most appropriate type is `StaticDraw`:

```cpp
m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
```

Then, the buffer itself has to be created and bound to the current context (for example, in the `initializeGL()` function):

```cpp
m_vertexBuffer.create();
m_vertexBuffer.bind();
```

The next step is to actually allocate some memory for the buffer and initialize it:

```cpp
m_vertexBuffer.allocate(m_data.constData(),
                        m_data.count() * sizeof(ScenePoint));
```

To change data in the buffer, there are two options. First, you can attach the buffer to the application's memory space, using a call to `map()` and then fill the data, using a returned pointer:

```cpp
ScenePoint *buffer = static_cast<ScenePoint*>(
    vbo.map(QOpenGLBuffer::WriteOnly));
assert(buffer != nullptr);
for(int i = 0; i < vbo.size(); ++i) {
    buffer[i] = ...;
}
vbo.unmap(); 
```

An alternative approach is to write to the buffer directly, using `write()`:

```cpp
vbo.write(0, m_data.constData(), m_data.size() * sizeof(ScenePoint));
```

Finally, the buffer can be used in the shader program in a way similar to an attribute array:

```cpp
vbo.bind();
m_shader.setAttributeBuffer("Vertex", GL_FLOAT,
                            0, 3, sizeof(ScenePoint));
m_shader.enableAttributeArray("Vertex");
m_shader.setAttributeBuffer("Normal", GL_FLOAT,
                            sizeof(QVector3D), 3, sizeof(ScenePoint));
m_shader.enableAttributeArray("Normal");
```

The result is that all the data is uploaded to the GPU once and then used as needed by the current shader program or other OpenGL call-supporting buffer objects.

# Using multiple OpenGL versions

Earlier in this chapter, we discussed a family of `QOpenGLFunctions` classes that provide access to OpenGL functions included in a specific OpenGL profile. If your whole application can use one profile, you can just select the appropriate Qt class and use it. However, sometimes you don't want the application to shut down completely if the requested profile is not supported on the current system. Instead, you can relax your requirements and use an older OpenGL version and provide simplified but still working rendering for systems that don't support the new profile. In Qt, you can implement such an approach using `QOpenGLContext::versionFunctions()`:

```cpp
class MyWindow : public QOpenGLWindow {
protected:
    QOpenGLFunctions_4_5_Core *glFunctions45;
    QOpenGLFunctions_3_3_Core *glFunctions33;
    void initializeGL()
    {
        glFunctions33 = context()->versionFunctions<QOpenGLFunctions_3_3_Core>();
        glFunctions45 = context()->versionFunctions<QOpenGLFunctions_4_5_Core>();
    }
    void paintGL() {
        if (glFunctions45) {
            // OpenGL 4.5 rendering
            // glFunctions45->...
        } else if (glFunctions33) {
            // OpenGL 3.3 rendering
            // glFunctions33->...
        } else {
            qFatal("unsupported OpenGL version");
        }
    }
};
```

In the `initializeGL()` function, we try to request wrapper objects for multiple OpenGL versions. If the requested version is not currently available, `versionFunctions()` will return `nullptr`. In the `paintGL()` function, we use the best available version to perform the actual rendering.

Next, you can use the `QSurfaceFormat` class to specify the OpenGL version and profile you want to use:

```cpp
MyWindow window;
QSurfaceFormat format = window.format();
format.setVersion(4, 0);
format.setProfile(QSurfaceFormat::CoreProfile);
window.setFormat(format);
window.show();
```

By requesting the core profile, you can ensure that old deprecated functionality will not be available in our application.

# Offscreen rendering

Sometimes, it is useful to render an OpenGL scene not to the screen but to some image that can be later processed externally or used as a texture in some other part of rendering. For that, the concept of **Framebuffer Objects** (**FBO**) was created. An FBO is a rendering surface that behaves like the regular device frame buffer, with the only exception that the resulting pixels do not land on the screen. An FBO target can be bound as a texture in an existing scene or dumped as an image to regular computer memory. In Qt, such an entity is represented by a `QOpenGLFramebufferObject` class.

Once you have a current OpenGL context, you can create an instance of `QOpenGLFramebufferObject`, using one of the available constructors. A mandatory parameter to pass is the size of the canvas (either as a `QSize` object or as a pair of integers describing the width and height of the frame). Different constructors accept other parameters, such as the type of texture the FBO is to generate or a set of parameters encapsulated in `QOpenGLFramebufferObjectFormat`.

When the object is created, you can issue a `bind()` call on it, which switches the OpenGL pipeline to render to the FBO instead of the default target. A complementary method is `release()`, which restores the default rendering target. Afterward, the FBO can be queried to return the ID of the OpenGL texture (using the `texture()` method) or to convert the texture to `QImage` (by invoking `toImage()`).

# Vulkan in Qt applications

OpenGL has undergone significant changes as graphics cards hardware has evolved. Many old parts of OpenGL API are now deprecated, and even up-to-date API is not ideal for utilizing the capabilities of modern hardware. Vulkan was designed as an attempt to create an API more suitable for this purpose.

Vulkan is a new API that can be used instead of OpenGL to perform hardware-accelerated rendering and computation. While Vulkan is more verbose and complex than OpenGL, it closely represents the actual interaction between CPU and GPU. This allows Vulkan users to achieve better control over utilizing GPU resources, which can lead to better performance. The first stable version of Vulkan API was released in 2016.

While Vulkan is a cross-platform solution, a Vulkan application still needs to contain a bit of platform-specific code, mainly related to window creation and event handling. Since Version 5.10, Qt provides a way to use Vulkan along with Qt's existing window and event infrastructure. You still retain full access to the original Vulkan API for rendering, but, at the same time, you can use the already familiar Qt API for everything else.

As with OpenGL, we will not give an in-depth guide of Vulkan here. We will only provide simple examples and cover the interaction between Qt and Vulkan. If you need more information about Vulkan, you can refer to its official page at [https://www.khronos.org/vulkan/](https://www.khronos.org/vulkan/).

# Preparing the developing environment

Before you can start developing games with Vulkan and Qt, you need to make a few preparations. First, you need to install the Vulkan SDK. To do that, head to [https://www.lunarg.com/vulkan-sdk/](https://www.lunarg.com/vulkan-sdk/), download a file for your operating system, and execute or unpack it. Examine the `index.html` file in the `doc` subdirectory in the installation folder to see whether you need to perform any additional actions.

Next, you need a Qt build with Vulkan support; it must be Qt 5.10 or later. If you have installed the most recent version available through the installer, it may already be suitable.

To check whether your Qt version has Vulkan support, create a new Qt Console Application, ensure that you select the kit corresponding to the most recently installed Qt version. The Vulkan SDK also requires you to set some environment variables, such as `VULKAN_SDK`, `PATH`, `LD_LIBRARY_PATH`, and `VK_LAYER_PATH` (exact names and values can depend on the operating system, so refer to the SDK documentation). You can edit environment variables for your project by switching to Qt Creator's Projects pane and expanding the Build Environment section.

Put the following code in `main.cpp`:

```cpp
#include <QGuiApplication>
#include <vulkan/vulkan.h>
#include <QVulkanInstance>
int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    QVulkanInstance vulkan;
    return app.exec();
}
```

Additionally, adjust the project file so that we actually have a Qt GUI application instead of a console application:

```cpp
QT += gui
CONFIG += c++11
DEFINES += QT_DEPRECATED_WARNINGS
SOURCES += main.cpp
```

If the project builds successfully, your setup is complete.

If the compiler can't find the `vulkan`/`vulkan.h` header, then the Vulkan SDK was not installed properly or its headers are not located in the default include path. Check the Vulkan SDK documentation to see whether you have missed something. You can also switch to the Projects pane of Qt Creator and edit the build environment of the project to make the installed headers visible. Depending on the compiler, you may need to set the `INCLUDEPATH` or `CPATH` environment variable.

If you have a compile error corresponding to the `QVulkanInstance` header, you are using a Qt version prior to 5.10\. Ensure that you install a recent version and select the correct kit on the Projects pane of Qt Creator.

However, if the `QVulkanInstance` includes directive works, but the `QVulkanInstance` class is still not defined, it means that your Qt build lacks Vulkan support. In this case, first try to install the most recent version using the official installer, if you haven't done so already:

1.  Close Qt Creator
2.  Launch the **Maintenance Tool** executable from the Qt installation directory
3.  Select Add or remove components
4.  Select the most recent Desktop Qt version
5.  Confirm the changes

After the installation is done, re-open Qt Creator, switch to the Projects pane, and select the new kit for the project.

Unfortunately, at the time of writing, the Qt builds available through the official installer do not have Vulkan support. It's possible (and likely) that it will be enabled in the future versions.

If the `QVulkanInstance` class is still not recognized, you have to build Qt from sources. This process varies depending on the operating system and the Qt version, so we will not cover the details in the book. Go to the [http://doc.qt.io/qt-5/build-sources.html](http://doc.qt.io/qt-5/build-sources.html) page and follow the instructions corresponding to your operating system. If the Vulkan SDK is properly installed, the output of the `configure` command should contain Vulkan ... yes, indicating that Vulkan support is enabled. After you build Qt, open Qt Creator's options dialog and set up a Qt version and a kit, as described in [Chapter 2](d129202d-f982-4114-b69a-094d0a136fe9.xhtml), *Installation*.

Finally, select the new kit for the project on the Projects pane:

![](img/9342b3d0-8bf3-4324-b01c-ce58faa85494.png)

If you've done everything correctly, the project should now build and execute successfully.

# Vulkan instance, window, and renderer

Before we start creating our first minimal Vulkan application, let's get familiar with the Qt classes we'll need for the task.

Unlike OpenGL, Vulkan doesn't have a global state. Interaction with Vulkan starts with the instance object represented by the `VkInstance` type. An application usually creates a single `VkInstance` object that contains the application-wide state. All other Vulkan objects can only be created from the instance object. In Qt, the corresponding class is `QVulkanInstance`. This class provides a convenient way to configure Vulkan and then initialize it with the given configuration. You can also use its `supportedExtensions()` and `supportedLayers()` functions to query supported features before using them. After the configuration is done, you should call the `create()` function that actually triggers loading Vulkan library and creating a `VkInstance` object. If this function returns `true`, the Vulkan instance object is ready to be used.

The next step is to create a window capable of Vulkan rendering. This is done by subclassing the `QVulkanWindow` class. Similar to `QOpenGLWindow`, `QVulkanWindow` extends `QWindow` and provides functionality required for utilizing Vulkan capabilities as well as some convenience functions. You can also use virtual functions inherited from `QWindow` to handle any events dispatched by Qt's event system. However, subclasses of `QVulkanWindow` should not perform any actual rendering. This task is delegated to the `QVulkanWindowRenderer` class. The `QVulkanWindow::createRenderer()` virtual function will be called once after the window is first shown, and you should reimplement this function to return your renderer object.

Now, about the renderer itself: `QVulkanWindowRenderer` is a simple class containing nothing more than a set of virtual functions. You can create your own renderer by subclassing `QVulkanWindowRenderer` and re-implementing the only pure virtual function called `startNextFrame()`. This function will be called when the drawing of the next frame is requested. You can perform all required drawing operations in this function and end it with a call to `QVulkanWindow::frameReady()` to indicate that the drawing is complete. You can also re-implement other virtual functions of the renderer. The most useful of them are `initResources()` and `releaseResources()`, which allow you to create required resources, store them in private members of your renderer class, and then destroy them when necessary.

These three classes define the basic structure of your Vulkan application. Let's see them in action.

# Time for action – Creating the minimal Vulkan project

We've already created a project while testing the developing environment. Now let's add two new classes to the project. One class named `MyWindow` should be derived from `QVulkanWindow`, and the other class named `MyRenderer` should be derived from `QVulkanWindowRenderer`. Implement the window's `createRenderer()` virtual function:

```cpp
QVulkanWindowRenderer *MyWindow::createRenderer() {
    return new MyRenderer(this);
}
```

Add the `QVulkanWindow *m_window` private field to the renderer class. Implement the constructor to initialize this field and override the `startNextFrame()` virtual function, as shown:

```cpp
MyRenderer::MyRenderer(QVulkanWindow *window)
{
    m_window = window;
}
void MyRenderer::startNextFrame() {
    m_window->frameReady();
}
```

Finally, edit the `main()` function:

```cpp
int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    QVulkanInstance vulkan;
    if (!vulkan.create()) {
        qFatal("Failed to create Vulkan instance: %d", vulkan.errorCode()); 
    }
    MyWindow window;
    window.resize(1024, 768);
    window.setVulkanInstance(&vulkan);
    window.show();
    return app.exec();
}
```

When you compile and run the project, a blank window with a black background should appear.

# What just happened?

We've created a window that will be rendered using Vulkan. The `main()` function initializes Vulkan, creates a window, passes the instance object to the window, and shows it on the screen. As usual, the final call to `exec()` starts Qt's event loop. When the window is shown, Qt will call the `createRenderer()` function on the window and a new renderer object will be created in your implementation of this function. The renderer is attached to the window and will automatically be deleted along with it, so there is no need to delete it manually. Each time the window needs to be painted, Qt will call the renderer's `startNextFrame()` function. We don't perform any painting yet, so the window remains blank.

It's important that the drawing of every frame ends with a call to `frameReady()`. Until this function is called, processing of the frame cannot be completed. However, it's not required to call this function directly from the `startNextFrame()` function. You can delay this call if you need, for example, to wait for calculations to complete in a separate thread.

Similar to how `paintEvent()` works, `startNextFrame()` will not be called continuously by default. It will only be called once after showing the window. It will also be called each time a part of the window is exposed (for example, as a result of moving a window or restoring a minimized window). If you need to render a dynamic scene continuously, call `m_window->requestUpdate()` after calling `m_window->frameReady()`.

# Using Vulkan types and functions

We can let Qt handle loading the Vulkan library and resolving functions for us. It works similar to the `QOpenGLFunctions` set of classes. Qt provides two functions classes for Vulkan:

*   The `QVulkanFunctions` class provides access to the Vulkan functions that are not device-specific
*   The `QVulkanDeviceFunctions` class provides functions that work on a specific `VkDevice`

You can obtain these objects by calling the `functions()` and `deviceFunctions(VkDevice device)` methods of the `QVulkanInstance` class, respectively. You will usually use the device functions a lot in the renderer, so a common pattern is to add the `QVulkanDeviceFunctions *m_devFuncs` private field to your renderer class and initialize it in the `initResources()` virtual function:

```cpp
void MyRenderer::initResources()
{
    VkDevice device = m_window->device();
    m_devFuncs = m_window->vulkanInstance()->deviceFunctions(device);
    //...
}
```

Now you can use `m_devFuncs` to access the Vulkan API functions. We won't use them directly, so we don't need to figure out how to link against the Vulkan library on each platform. Qt does this job for us.

As for structures, unions, and typedefs, we can use them directly without worrying about the platform details. It's enough to have the Vulkan SDK headers present in the system.

# Time for action – Drawing with a dynamic background color

Let's see how we can use the Vulkan API in our Qt project to change the background color of the window. We'll cycle through all possible hues of the color while retaining constant saturation and lightness. This may sound complicated when you think about a color in RGB space, but it's actually very easy if you work with the HSL (Hue, Saturation, Lightness) color model. Luckily, `QColor` supports multiple color models, including HSL.

First, add and initialize the `m_devFuncs` private field, as just shown. Next, add the `float m_hue` private field that will hold the current hue of the background color. Set its initial value to zero. We can now start writing our `startNextFrame()` function that will do all the magic. Let's go through it piece by piece. First, we increment our `m_hue` variable and ensure that we don't go out of bounds; then, we use the `QColor::fromHslF()` function to construct a `QColor` value based on given hue, saturation, and lightness (each of them ranges from 0 to 1):

```cpp
void MyRenderer::startNextFrame()
{
    m_hue += 0.005f;
    if (m_hue > 1.0f) {
        m_hue = 0.0f;
    }
    QColor color = QColor::fromHslF(m_hue, 1, 0.5);
    //...
}
```

Next, we use this `color` variable to construct a `VkClearValue` array that we'll use for setting the background color:

```cpp
  VkClearValue clearValues[2];
  memset(clearValues, 0, sizeof(clearValues));
  clearValues[0].color = {
      static_cast<float>(color.redF()),
      static_cast<float>(color.greenF()),
      static_cast<float>(color.blueF()),
      1.0f
  };
  clearValues[1].depthStencil = { 1.0f, 0 };
```

To start a new render pass in Vulkan, we need to initialize a `VkRenderPassBeginInfo` structure. It requires a lot of data, but, luckily, `QVulkanWindow` provides most of the data for us. We just need to put it into the structure and use the `clearValues` array we set up earlier:

```cpp
VkRenderPassBeginInfo info;
memset(&info, 0, sizeof(info));
info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
info.renderPass = m_window->defaultRenderPass();
info.framebuffer = m_window->currentFramebuffer();
const QSize imageSize = m_window->swapChainImageSize();
info.renderArea.extent.width = imageSize.width();
info.renderArea.extent.height = imageSize.height();
info.clearValueCount = 2;
info.pClearValues = clearValues;
```

Finally, it's time to perform the rendering:

```cpp
VkCommandBuffer commandBuffer = m_window->currentCommandBuffer();
m_devFuncs->vkCmdBeginRenderPass(commandBuffer, &info,
                                 VK_SUBPASS_CONTENTS_INLINE);
m_devFuncs->vkCmdEndRenderPass(commandBuffer);
m_window->frameReady();
m_window->requestUpdate();
```

The `vkCmdBeginRenderPass()` Vulkan API function will begin the render pass, which will result in clearing the window with the color we've set. Since we don't have anything else to draw, we complete the render pass immediately using the `vkCmdEndRenderPass()` function. Then, we indicate that we've already done everything we want for this frame by calling the `frameReady()` function. This allows Qt to advance the rendering loop. As the final step, we request an update of the window to ensure that the new frame will be requested soon and the color animation will go on.

If you run the project now, you should see a window that constantly changes its background color:

![](img/e9241a73-f35a-4db6-bc40-79b7c45f456c.png)

We would love to show a more advanced example. However, even drawing a simple triangle in Vulkan usually requires a few hundred lines of code, because Vulkan requires you to explicitly set up a lot of things. While Qt provides a lot of helper classes for OpenGL rendering, it does not contain any similar classes that would help with Vulkan rendering or computation (as of Qt 5.10), so there is nothing specific to Qt in these tasks.

If you want to deepen your knowledge of Vulkan, you can study the documentation and tutorials present on its official website and the Vulkan SDK website. Qt also includes several good examples based on Vulkan, such as Hello Vulkan Triangle, Hello Vulkan Texture, and Hello Vulkan Cubes.

# Logs and validation

Qt automatically receives messages from the Vulkan library and puts them into Qt's own logging system. The critical errors will be passed to `qWarning()`, so they will appear in the application output by default. However, Qt also logs additional information that can be useful when debugging. This information is hidden by default, but you can make it visible by adding the following line to the `main()` function just after the construction of `QGuiApplication`:

```cpp
QLoggingCategory::setFilterRules(QStringLiteral("qt.vulkan=true"));
```

The Vulkan API does not perform any sanity checks by default. If you pass an invalid parameter to a Vulkan API function, the application may silently crash, or work inconsistently. However, you can enable **validation layers** for your Vulkan instance. They do not change the functionality of the API calls, but they enable additional checks when possible. It's a good idea to enable validation layers in a debug build. You can do that by calling `setLayers()` on the instance object before calling `create()`:

```cpp
vulkan.setLayers({ "VK_LAYER_LUNARG_standard_validation" });
```

Keep in mind that an attempt to request a currently unsupported layer or extension will be ignored by Qt.

Let's test the validation layers by inserting an invalid parameter to our code:

```cpp
info.renderArea.extent.width = -5; // invalid
```

When you run the application, Qt should print a warning to the application output:

```cpp
vkDebug: CORE: 4: Cannot execute a render pass with renderArea not within the bound of the framebuffer. RenderArea: x 0, y 0, width -5, height 768\. Framebuffer: width 1024, height 768.
```

If the warning does not appear, it means that the validation layers are not available or they failed to load. Check the application output for the presence of validation layers (they will be printed after the "Supported Vulkan instance layers" line) and any library loading errors. Ensure that you've set up the Vulkan SDK and the project's environment variables according to the documentation.

However, keep in mind that validation layers have a performance impact on your application. You should probably disable them in your final builds. You can also disable redirecting Vulkan's debug output to the Qt logging system, using the following code:

```cpp
QVulkanInstance vulkan;
vulkan.setFlags(QVulkanInstance::NoDebugOutputRedirect);
```

# Combining OpenGL or Vulkan with Qt Widgets

Sometimes you want to combine the powers of accelerated graphics and Qt Widgets. While OpenGL and Vulkan are great for rendering high-performance 2D and 3D scenes, the Qt Widgets module is far easier to use for creating user interfaces. Qt offers a few ways to combine them into a single powerful interface. This can be useful if your application depends heavily on widgets (for example, the 3D view is only one of the views in your application and is controlled using a bunch of other widgets surrounding the main view).

The first way is the `QWidget::createWindowContainer()` function. It takes an arbitrary `QWindow` and creates a `QWidget` that keeps the window within its bounds. That widget can be put into another widget and can be managed by a layout. While the window appears to be embedded into another window, it still remains a native window from the operating system's perspective, and any accelerated rendering will be performed directly on the window without a heavy performance impact. This approach has a few limitations, though. For example, the embedded window will always stack on top of other widgets. However, it's suitable in most cases.

Let's return to our OpenGL cube project and put it into a layout with an additional label:

```cpp
QWidget widget;
QVBoxLayout* layout = new QVBoxLayout(&widget);
layout->addWidget(new QLabel("Scene"), 0);
QWidget* container = QWidget::createWindowContainer(&window, &widget);
layout->addWidget(container, 1);
widget.resize(600, 600);
widget.show();
```

Instead of showing the OpenGL window, we created a widget and put the window into the layout of that widget:

![](img/a629f0d8-28a1-4045-9821-edbff584173f.png)

You can apply this approach to any `QWindow`, including Vulkan-based windows and Qt Quick windows, which we'll work with in subsequent chapters.

There is another way to solve the same task, but it only works with OpenGL. You can simply replace `QOpenGLWindow` with `QOpenGLWidget` to turn a window into a fully featured widget. The API of `QOpenGLWidget` (including virtual functions) is compatible with `QOpenGLWindow`, so it can act as a drop-in replacement. There are no limitations for the  stacking order, focus, or opacity of `QOpenGLWidget`. You can even mix the OpenGL rendering with `QPainter` operations. However, this solution has a performance cost. `QOpenGLWindow` renders directly to the given window, while `QOpenGLWidget` first renders to an offscreen buffer that is then rendered to the widget, so it will be slower.

# Pop quiz

Q1\. Which of the following programming languages is accepted by the `QOpenGLShader::compileSourceCode()` function?

1.  C
2.  C++
3.  GLSL

Q2\. Which virtual function of the `QOpenGLWindow` class should you implement to perform OpenGL painting?

1.  `paintGL()`
2.  `paintEvent()`
3.  `makeCurrent()`

Q3\. When should you delete the object of your `QVulkanWindowRenderer` subclass?

1.  In the destructor of the `QVulkanWindow` subclass
2.  After deleting the `QVulkanInstance` object
3.  Never

# Summary

In this chapter, we learned about using OpenGL and Vulkan graphics with Qt. With this knowledge, you can create hardware accelerated 2D and 3D graphics. We also explored Qt classes that simplify usage of these technologies in Qt applications. If you want to sharpen your OpenGL and Vulkan skills, you can study numerous books and articles focused on these topics. Qt provides very transparent access to hardware accelerated graphics, so adapting any pure OpenGL or Vulkan approaches for Qt should be easy. If you prefer to have a higher-level API for accelerated graphics, you should turn your attention to Qt Quick and Qt 3D. We will cover it in the last part of this book.

In the next chapter, you will learn to implement scripting in your game. This will make it more extensible and easier to modify. Scripting can also be used to enable modding in your game, allowing players to customize the gameplay how they want.