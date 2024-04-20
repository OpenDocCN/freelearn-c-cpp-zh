# *第六章*：构建抽象渲染器

本书侧重于动画，而不是渲染。然而，渲染动画模型是很重要的。为了避免陷入任何特定的图形 API 中，在本章中，您将在 OpenGL 之上构建一个抽象层。这将是一个薄的抽象层，但它将让您在后面的章节中处理动画，而无需执行任何特定于 OpenGL 的操作。

本章中您将实现的抽象渲染器非常轻量。它没有很多功能，只有您需要显示动画模型的功能。这应该使得将渲染器移植到其他 API 变得简单。

在本章结束时，您应该能够使用您创建的抽象渲染代码在窗口中渲染一些调试几何体。在更高的层次上，您将学到以下内容：

+   如何创建着色器

+   如何在缓冲区中存储网格数据

+   如何将这些缓冲区绑定为着色器属性

+   如何向着色器发送统一数据

+   如何使用索引缓冲区进行渲染

+   如何加载纹理

+   基本的 OpenGL 概念

+   创建和使用简单的着色器

# 技术要求

对 OpenGL 的一些了解将使本章更容易理解。OpenGL、光照模型和着色器技巧不在本书的范围之内。有关这些主题的更多信息，请访问[`learnopengl.com/`](https://learnopengl.com/)。

# 使用着色器

抽象层中最重要的部分是`Shader`类。要绘制某物，您必须绑定一个着色器并将一些属性和统一附加到它上。着色器描述了被绘制的东西应该如何变换和着色，而属性定义了正在被绘制的内容。

在本节中，您将实现一个`Shader`类，它可以编译顶点和片段着色器。`Shader`类还将返回统一和属性索引。

## 着色器类声明

在实现`Shader`类时，您需要声明几个受保护的辅助函数。这些函数将保持类的公共 API 清晰；它们用于诸如将文件读入字符串或调用 OpenGL 代码来编译着色器的操作：

1.  创建一个新文件来声明`Shader`类，命名为`Shader.h`。`Shader`类应该有一个指向 OpenGL 着色器对象的句柄，以及属性和统一索引的映射。这些字典有一个字符串作为键（属性或统一的名称）和一个`unsigned int`作为值（统一或属性的索引）：

```cpp
    class Shader {
    private:
        unsigned int mHandle;
        std::map<std::string, unsigned int> mAttributes;
        std::map<std::string, unsigned int> mUniforms;
    ```

1.  `Shader`类的复制构造函数和赋值运算符应该被禁用。`Shader`类不打算通过值进行复制，因为它持有一个 GPU 资源的句柄：

```cpp
    private:
        Shader(const Shader&);
        Shader& operator=(const Shader&);
    ```

1.  接下来，您需要在`Shader`类中声明辅助函数。`ReadFile`函数将文件内容读入`std::string`中。`CompileVertexShader`和`CompileFragmentShader`函数编译着色器源代码并返回 OpenGL 句柄。`LinkShader`函数将两个着色器链接成一个着色器程序。`PopulateAttribute`和`PopulateUniform`函数将填充属性和统一字典：

```cpp
    private:
        std::string ReadFile(const std::string& path);
        unsigned int CompileVertexShader(
                         const std::string& vertex);
        unsigned int CompileFragmentShader(
                         const std::string& fragment);
        bool LinkShaders(unsigned int vertex, 
                         unsigned int fragment);
        void PopulateAttributes();
        void PopulateUniforms();
    ```

1.  类的默认构造函数将创建一个空的`Shader`对象。重载构造函数将调用`Load`方法，从文件加载着色器并编译它们。析构函数将释放`Shader`类持有的 OpenGL 着色器句柄：

```cpp
    public:
        Shader();
        Shader(const std::string& vertex, 
               const std::string& fragment);
        ~Shader();
        void Load(const std::string& vertex, 
                  const std::string& fragment);
    ```

1.  在使用着色器之前，需要使用`Bind`函数绑定它。同样，在不再使用时，可以使用`UnBind`函数解绑它。`GetAttribute`和`GetUniform`函数在适当的字典中执行查找。`GetHandle`函数返回着色器的 OpenGL 句柄：

```cpp
        void Bind();
        void UnBind();
        unsigned int GetAttribute(const std::string& name);
        unsigned int GetUniform(const std::string& name);
        unsigned int GetHandle();
    };
    ```

现在`Shader`类声明完成后，您将在下一节中实现它。

## 实现着色器类

创建一个新文件`Shader.cpp`，来实现`Shader`类。`Shader`类的实现几乎将所有实际的 OpenGL 代码隐藏在调用者之外。因为大多数 OpenGL 调用都是通过这种方式抽象的，在后面的章节中，您只需要调用抽象层，而不是直接调用 OpenGL 函数。

本书中始终使用统一数组。当在着色器中遇到统一数组（例如`modelMatrices[120]`），`glGetActiveUniform`返回的统一名称是数组的第一个元素。在这个例子中，那将是`modelMatrices[0]`。当遇到统一数组时，您希望循环遍历所有数组索引，并为每个元素获取显式的统一索引，但您还希望存储没有任何下标的统一名称：

1.  两个`Shader`构造函数必须通过调用`glCreateProgram`创建一个新的着色器程序句柄。接受两个字符串的构造函数变体调用`Load`函数处理这些字符串。由于`mHandle`始终是一个程序句柄，析构函数需要删除该句柄：

```cpp
    Shader::Shader() {
        mHandle = glCreateProgram();
    }
    Shader::Shader(const std::string& vertex, 
                   const std::string& fragment) {
        mHandle = glCreateProgram();
        Load(vertex, fragment);
    }
    Shader::~Shader() {
        glDeleteProgram(mHandle);
    }
    ```

1.  `ReadFile`辅助函数使用`std::ifstream`将文件转换为字符串，以读取文件的内容到`std::stringstream`中。字符串流可用于将文件内容作为字符串返回：

```cpp
    std::string Shader::ReadFile(const std::string& path) {
        std::ifstream file;
        file.open(path);
        std::stringstream contents;
        contents << file.rdbuf();
        file.close();
        return contents.str();
    }
    ```

1.  `CompileVertexShader`函数是用于编译 OpenGL 顶点着色器的样板代码。首先，使用`glCreateShader`创建着色器对象，然后使用`glShaderSource`为着色器设置源。最后，使用`glCompileShader`编译着色器。使用`glGetShaderiv`检查错误：

```cpp
    unsigned int Shader::CompileVertexShader(
                                   const string& vertex) {
        unsigned int v = glCreateShader(GL_VERTEX_SHADER);
        const char* v_source = vertex.c_str();
        glShaderSource(v, 1, &v_source, NULL);
        glCompileShader(v);
        int success = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(v, 512, NULL, infoLog);
            std::cout << "Vertex compilation failed.\n";
            std::cout << "\t" << infoLog << "\n";
            glDeleteShader(v);
            return 0;
        };
        return v;
    }
    ```

1.  `CompileFragmentShader`函数与`CompileVertexShader`函数几乎完全相同。唯一的真正区别是`glCreateShader`的参数，表明您正在创建一个片段着色器，而不是顶点着色器：

```cpp
    unsigned int Shader::CompileFragmentShader(
                              const std::string& fragment) {
        unsigned int f = glCreateShader(GL_FRAGMENT_SHADER);
        const char* f_source = fragment.c_str();
        glShaderSource(f, 1, &f_source, NULL);
        glCompileShader(f);
        int success = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(f, 512, NULL, infoLog);
            std::cout << "Fragment compilation failed.\n";
            std::cout << "\t" << infoLog << "\n";
            glDeleteShader(f);
            return 0;
        };
        return f;
    }
    ```

1.  `LinkShaders`辅助函数也是样板。将着色器附加到构造函数创建的着色器程序句柄。通过调用`glLinkProgram`链接着色器，并使用`glGetProgramiv`检查错误。一旦着色器被链接，您只需要程序；可以使用`glDeleteShader`删除各个着色器对象：

```cpp
    bool Shader::LinkShaders(unsigned int vertex, 
                             unsigned int fragment) {
        glAttachShader(mHandle, vertex);
        glAttachShader(mHandle, fragment);
        glLinkProgram(mHandle);
        int success = 0;
        glGetProgramiv(mHandle, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(mHandle, 512, NULL, infoLog);
            std::cout << "ERROR: Shader linking failed.\n";
            std::cout << "\t" << infoLog << "\n";
            glDeleteShader(vertex);
            glDeleteShader(fragment);
            return false;
        }
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        return true;
    }
    ```

1.  `PopulateAttributes`函数枚举存储在着色器程序中的所有属性，然后将它们存储为键值对，其中键是属性的名称，值是其位置。您可以使用`glGetProgramiv`函数计算着色器程序中活动属性的数量，将`GL_ACTIVE_ATTRIBUTES`作为参数名称传递。然后，通过索引循环遍历所有属性，并使用`glGetActiveAttrib`获取每个属性的名称。最后，调用`glGetAttribLocation`获取每个属性的位置：

```cpp
    void Shader::PopulateAttributes() {
        int count = -1;
        int length;
        char name[128];
        int size;
        GLenum type;
        glUseProgram(mHandle);
        glGetProgramiv(mHandle, GL_ACTIVE_ATTRIBUTES, 
                       &count);
        for (int i = 0; i < count; ++i) {
            memset(name, 0, sizeof(char) * 128);
            glGetActiveAttrib(mHandle, (GLuint)i, 128, 
                              &length, &size, &type, name);
            int attrib = glGetAttribLocation(mHandle, name);
            if (attrib >= 0) {
                mAttributes[name] = attrib;
            }
        }
        glUseProgram(0);
    }
    ```

1.  `PopulateUniforms`辅助函数与`PopulateAttributes`辅助函数非常相似。`glGetProgramiv`需要以`GL_ACTIVE_UNIFORMS`作为参数名称，并且您需要调用`glGetActiveUniform`和`glGetUniformLocation`：

```cpp
    void Shader::PopulateUniforms() {
        int count = -1;
        int length;
        char name[128];
        int size;
        GLenum type;
        char testName[256];
        glUseProgram(mHandle);
        glGetProgramiv(mHandle, GL_ACTIVE_UNIFORMS, &count);
        for (int i = 0; i < count; ++i) {
            memset(name, 0, sizeof(char) * 128);
            glGetActiveUniform(mHandle, (GLuint)i, 128, 
                               &length, &size, &type, name);
            int uniform=glGetUniformLocation(mHandle, name);
            if (uniform >= 0) { // Is uniform valid?
    ```

1.  当遇到有效的统一时，您需要确定该统一是否是一个数组。为此，在统一名称中搜索数组括号（`[`）。如果找到括号，则该统一是一个数组：

```cpp
    std::string uniformName = name;
    // if name contains [, uniform is array
    std::size_t found = uniformName.find('[');
    if (found != std::string::npos) {
    ```

1.  如果遇到一个统一数组，从`[`开始擦除字符串中的所有内容。这将使您只剩下统一的名称。然后，进入一个循环，尝试通过将`[ + index + ]`附加到统一名称来检索数组中的每个索引。一旦找到第一个无效的索引，就打破循环：

```cpp
    uniformName.erase(uniformName.begin() + 
         found, uniformName.end());
         unsigned int uniformIndex = 0;
         while (true) {
               memset(testName,0,sizeof(char)*256);
                   sprintf(testName, "%s[%d]", 
                               uniformName.c_str(), 
                               uniformIndex++);
                       int uniformLocation = 
                               glGetUniformLocation(
                               mHandle, testName);
                       if (uniformLocation < 0) {
                          break;
                       }
                       mUniforms[testName]=uniformLocation;
                    }
                }
    ```

1.  此时，`uniformName`包含统一的名称。如果该统一是一个数组，则名称的`[0]`部分已被移除。按名称将统一索引存储在`mUniforms`中：

```cpp
                mUniforms[uniformName] = uniform;
            }
        }
        glUseProgram(0);
    }
    ```

1.  最后一个辅助函数是`Load`函数，负责加载实际的着色器。此函数接受两个字符串，可以是文件名或内联着色器定义。一旦读取了着色器，调用`Compile`、`Link`和`Populate`辅助函数来加载着色器：

```cpp
    void Shader::Load(const std::string& vertex, 
                      const std::string& fragment) {
        std::ifstream f(vertex.c_str());
        bool vertFile = f.good();
        f.close();
        f = std::ifstream(vertex.c_str());
        bool fragFile = f.good();
        f.close();
        std::string v_source = vertex;
        if (vertFile) {
            v_source = ReadFile(vertex);
        }
        std::string f_source = fragment;
        if (fragFile) {
            f_source = ReadFile(fragment);
        }
        unsigned int vert = CompileVertexShader(v_source);
        unsigned int f = CompileFragmentShader(f_source);
        if (LinkShaders(vert, frag)) {
            PopulateAttributes();
            PopulateUniforms();
        }
    }
    ```

1.  `Bind`函数需要将当前着色器程序设置为活动状态，而`UnBind`应确保没有活动的`Shader`对象。`GetHandle`辅助函数返回`Shader`对象的 OpenGL 句柄：

```cpp
    void Shader::Bind() {
        glUseProgram(mHandle);
    }
    void Shader::UnBind() {
        glUseProgram(0);
    }
    unsigned int Shader::GetHandle() {
        return mHandle;
    }
    ```

1.  最后，您需要一种方法来检索属性和统一的绑定槽。`GetAttribute`函数将检查给定的属性名称是否存在于属性映射中。如果存在，则返回表示它的整数。如果没有，则返回`0`。`0`是有效的属性索引，因此在出现错误的情况下，还会记录错误消息：

```cpp
    unsigned int Shader::GetAttribute(
                            const std::string& name) {
        std::map<std::string, unsigned int>::iterator it =
                                    mAttributes.find(name);
        if (it == mAttributes.end()) {
            cout << "Bad attrib index: " << name << "\n";
            return 0;
        }
        return it->second;
    }
    ```

1.  `GetUniform`函数的实现几乎与`GetAttribute`函数相同，只是它不是在属性映射上工作，而是在统一映射上工作：

```cpp
    unsigned int Shader::GetUniform(const std::string& name){
        std::map<std::string, unsigned int>::iterator it =
                                      mUniforms.find(name);
        if (it == mUniforms.end()) {
            cout << "Bad uniform index: " << name << "\n";
            return 0;
        }
        return it->second;
    }
    ```

`Shader`类有方法来检索统一和属性的索引。在下一节中，您将开始实现一个`Attribute`类来保存传递给着色器的顶点数据。

# 使用缓冲区（属性）

属性是图形管道中的每个顶点数据。一个顶点由属性组成。例如，一个顶点有一个位置和一个法线，这两个都是属性。最常见的属性如下：

+   位置：通常在局部空间中

+   法线：顶点指向的方向

+   UV 或纹理坐标：纹理上的标准化（*x*，*y*）坐标

+   颜色：表示顶点颜色的`vector3`

属性可以具有不同的数据类型。在本书中，您将实现对整数、浮点数和矢量属性的支持。对于矢量属性，将支持二维、三维和四维向量。

## `Attribute`类声明

创建一个新文件`Attribute.h`。`Attribute`类将在这个新文件中声明。`Attribute`类将被模板化。这将确保如果一个属性被认为是`vec3`，您不能意外地将`vec2`加载到其中：

1.  属性类将包含两个成员变量，一个用于 OpenGL 属性句柄，一个用于计算`Attribute`类包含的数据量。由于属性数据存储在 GPU 上，您不希望有多个句柄指向相同的数据，因此应禁用复制构造函数和`赋值运算符`：

```cpp
    template<typename T>
    class Attribute {
    protected:
        unsigned int mHandle;
        unsigned int mCount;
    private:
        Attribute(const Attribute& other);
        Attribute& operator=(const Attribute& other);
    ```

1.  `SetAttribPointer`函数很特殊，因为它需要为每种支持的属性类型实现一次。这将在`.cpp`文件中明确完成：

```cpp
    void SetAttribPointer(unsigned int slot);
    ```

1.  将`Attribute`类的构造函数和析构函数声明为公共函数：

```cpp
    public:
        Attribute();
        ~Attribute();
    ```

1.  `Attribute`类需要一个`Set`函数，它将数组数据上传到 GPU。数组中的每个元素表示一个顶点的属性。我们需要一种从着色器定义的绑定槽中绑定和解绑属性的方法，以及属性的计数和句柄的访问器：

```cpp
        void Set(T* inputArray, unsigned int arrayLength);
        void Set(std::vector<T>& input);
        void BindTo(unsigned int slot);
        void UnBindFrom(unsigned int slot);
        unsigned int Count();
        unsigned int GetHandle();
    };
    ```

现在您已经声明了`Attribute`类，您将在下一节中实现它。

## 实现`Attribute`类

创建一个新文件`Attribtue.cpp`。您将在此文件中实现`Attribute`类如下：

1.  `Attribute`类是模板的，但它的函数都没有标记为内联。每种属性类型的模板特化将存在于`Attribute.cpp`文件中。为整数、浮点数、`vec2`、`vec3`、`vec4`和`ivec4`类型添加特化：

```cpp
    template Attribute<int>;
    template Attribute<float>;
    template Attribute<vec2>;
    template Attribute<vec3>;
    template Attribute<vec4>;
    template Attribute<ivec4>;
    ```

1.  构造函数应生成一个 OpenGL 缓冲区并将其存储在`Attribute`类的句柄中。析构函数负责释放`Attribute`类持有的句柄：

```cpp
    template<typename T>
    Attribute<T>::Attribute() {
        glGenBuffers(1, &mHandle);
        mCount = 0;
    }
    template<typename T>
    Attribute<T>::~Attribute() {
        glDeleteBuffers(1, &mHandle);
    }
    ```

1.  `Attribute`类有两个简单的 getter，一个用于检索计数，一个用于检索 OpenGL 句柄。计数表示总共有多少个属性：

```cpp
    template<typename T>
    unsigned int Attribute<T>::Count() {
        return mCount;
    }
    template<typename T>
    unsigned int Attribute<T>::GetHandle() {
        return mHandle;
    }
    ```

1.  `Set`函数接受一个数组和一个长度。然后绑定`Attribute`类持有的缓冲区，并使用`glBufferData`填充缓冲区数据。有一个方便的`Set`函数，它接受一个向量引用而不是数组。它调用实际的`Set`函数：

```cpp
    template<typename T>
    void Attribute<T>::Set(T* inputArray, 
                           unsigned int arrayLength) {
        mCount = arrayLength;
        unsigned int size = sizeof(T);
        glBindBuffer(GL_ARRAY_BUFFER, mHandle);
        glBufferData(GL_ARRAY_BUFFER, size * mCount, 
                     inputArray, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    template<typename T>
    void Attribute<T>::Set(std::vector<T>& input) {
        Set(&input[0], (unsigned int)input.size());
    }
    ```

1.  `SetAttribPointer`函数包装了`glVertesAttribPointer`或`glVertesAttribIPointer`。根据`Attribute`类的类型，参数和要调用的函数是不同的。为了消除任何歧义，为所有支持的模板类型提供显式实现。首先实现`int`、`ivec4`和`float`类型：

```cpp
    template<>
    void Attribute<int>::SetAttribPointer(unsigned int s) {
       glVertexAttribIPointer(s, 1, GL_INT, 0, (void*)0);
    }
    template<>
    void Attribute<ivec4>::SetAttribPointer(unsigned int s){
       glVertexAttribIPointer(s, 4, GL_INT, 0, (void*)0);
    }
    template<>
    void Attribute<float>::SetAttribPointer(unsigned int s){
       glVertexAttribPointer(s,1,GL_FLOAT,GL_FALSE,0,0);
    }
    ```

1.  接下来实现`vec2`、`vec3`和`vec4`类型。这些都与`float`类型非常相似。唯一的区别是`glVertexAttribPointer`的第二个参数：

```cpp
    template<>
    void Attribute<vec2>::SetAttribPointer(unsigned int s) {
       glVertexAttribPointer(s,2,GL_FLOAT,GL_FALSE,0,0);
    }
    template<>
    void Attribute<vec3>::SetAttribPointer(unsigned int s){
       glVertexAttribPointer(s,3,GL_FLOAT,GL_FALSE,0,0);
    }
    template<>
    void Attribute<vec4>::SetAttribPointer(unsigned int s){
       glVertexAttribPointer(s,4,GL_FLOAT,GL_FALSE,0,0);
    }
    ```

1.  `Attribute`类的最后两个函数需要将属性绑定到`Shader`类中指定的槽位，并解除绑定。由于`Attribute`类的模板类型不同，`Bind`将调用`SetAttribPointer`辅助函数：

```cpp
    template<typename T>
    void Attribute<T>::BindTo(unsigned int slot) {
        glBindBuffer(GL_ARRAY_BUFFER, mHandle);
        glEnableVertexAttribArray(slot);
        SetAttribPointer(slot);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    template<typename T>
    void Attribute<T>::UnBindFrom(unsigned int slot) {
        glBindBuffer(GL_ARRAY_BUFFER, mHandle);
        glDisableVertexAttribArray(slot);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    ```

`Attribute`数据每个顶点都会发生变化。您需要设置另一种类型的数据：uniforms。与属性不同，uniforms 在着色器程序执行期间保持不变。您将在下一节中实现 uniforms。

# 使用 uniforms

与属性不同，uniforms 是常量数据；它们只设置一次。uniform 的值对所有处理的顶点保持不变。uniforms 可以创建为数组，这是您将在后续章节中用来实现网格蒙皮的功能。

与`Attribute`类一样，`Uniform`类也将是模板化的。但与属性不同，永远不会有`Uniform`类的实例。它只需要公共静态函数。对于每种 uniform 类型，有三个函数：一个用于设置单个 uniform 值，一个用于设置一组 uniform 值，一个便利函数用于设置一组值，但使用向量作为输入。

## Uniform 类声明

创建一个新文件，`Uniform.h`。您将在这个新文件中实现`Uniform`类。`Uniform`类永远不会被实例化，因为不会有这个类的实例。禁用构造函数和复制构造函数、赋值运算符和析构函数。该类将具有三个静态`Set`函数的重载。`Set`函数需要为每种模板类型指定：

```cpp
template <typename T>
class Uniform {
private:
  Uniform();
  Uniform(const Uniform&);
  Uniform& operator=(const Uniform&);
  ~Uniform();
public:
  static void Set(unsigned int slot, const T& value);
  static void Set(unsigned int slot,T* arr,unsigned int len);
  static void Set(unsigned int slot, std::vector<T>& arr);
};
```

您刚刚完成了`Uniform`类的声明。在下一节中，您将开始实现`Uniform`类。

## 实现 Uniform 类

创建一个新文件，`Uniform.cpp`。您将在这个新文件中实现`Uniform`类。与`Attribute`类一样，`Uniform`类也是模板化的。

在 OpenGL 中，uniforms 是使用`glUniform***`系列函数设置的。有不同的函数用于整数、浮点数、向量、矩阵等。您希望为每种类型的`Set`方法提供实现，但避免编写几乎相同的代码。

为了避免编写几乎相同的代码，您将声明一个`#define`宏。这个宏将接受三个参数——要调用的 OpenGL 函数，Uniform 类的模板类型和 OpenGL 函数的数据类型：

1.  添加以下代码以定义支持的 uniform 类型的模板规范：

```cpp
    template Uniform<int>;
    template Uniform<ivec4>;
    template Uniform<ivec2>;
    template Uniform<float>;
    template Uniform<vec2>;
    template Uniform<vec3>;
    template Uniform<vec4>;
    template Uniform<quat>;
    template Uniform<mat4>;
    ```

1.  您只需要为每种类型实现一个`Set`方法，即接受数组和长度的方法。其他`Set`方法重载是为了方便起见。实现两个便利重载——一个用于设置单个 uniform，另一个用于设置向量。两个重载应该只调用`Set`函数：

```cpp
    template <typename T>
    void Uniform<T>::Set(unsigned int slot,const T& value){
        Set(slot, (T*)&value, 1);
    }
    template <typename T>
    void Uniform<T>::Set(unsigned int s,std::vector<T>& v){
        Set(s, &v[0], (unsigned int)v.size());
    }
    ```

1.  创建一个`UNIFORM_IMPL`宏。第一个参数是要调用的 OpenGL 函数，第二个是正在使用的结构类型，最后一个参数是相同结构的数据类型。`UNIFORM_IMPL`宏将这些信息组装成一个函数声明：

```cpp
    #define UNIFORM_IMPL(gl_func, tType, dType) \
    template<> void Uniform<tType>::Set(unsigned int slot,\
                       tType* data, unsigned int length) {\
        gl_func(slot, (GLsizei)length, (dType*)&data[0]); \
    }
    ```

1.  为每种 uniform 数据类型调用`UNIFORM_IMPL`宏以生成适当的`Set`函数。这种方法无法适用于`mat4`数据类型：

```cpp
    UNIFORM_IMPL(glUniform1iv, int, int)
    UNIFORM_IMPL(glUniform4iv, ivec4, int)
    UNIFORM_IMPL(glUniform2iv, ivec2, int)
    UNIFORM_IMPL(glUniform1fv, float, float)
    UNIFORM_IMPL(glUniform2fv, vec2, float)
    UNIFORM_IMPL(glUniform3fv, vec3, float)
    UNIFORM_IMPL(glUniform4fv, vec4, float)
    UNIFORM_IMPL(glUniform4fv, quat, float)
    ```

1.  矩阵的`Set`函数需要手动指定；否则，`UNIFORM_IMPL`宏将无法工作。这是因为`glUniformMatrix4fv`函数需要一个额外的布尔参数，询问矩阵是否应该被转置。将转置布尔值设置为`false`：

```cpp
    template<> void Uniform<mat4>::Set(unsigned int slot, 
            mat4* inputArray, unsigned int arrayLength) {
        glUniformMatrix4fv(slot, (GLsizei)arrayLength, 
                           false, (float*)&inputArray[0]);
    }
    ```

在本节中，你在统一的概念上构建了一个抽象层。在下一节中，你将实现类似属性的索引缓冲区。

# 使用索引缓冲区

索引缓冲区是一种属性。与属性不同，索引缓冲区绑定到`GL_ELEMENT_ARRAY_BUFFER`，可以用于绘制基本图元。因此，你将在它们自己的类中实现索引缓冲区，而不是重用`Attribute`类。

## IndexBuffer 类声明

创建一个新文件，`IndexBuffer.h`。你将在这个新文件中添加`IndexBuffer`类的声明。像`Attribute`对象一样，`IndexBuffer`将包含一个 OpenGL 句柄和一个计数，同时有 getter 函数。

为了避免多个`IndexBuffer`对象引用同一个 OpenGL 缓冲区，需要禁用复制构造函数和赋值运算符。`Set`函数接受一个无符号整数数组和数组的长度，但也有一个方便的重载，接受一个向量：

```cpp
class IndexBuffer {
public:
    unsigned int mHandle;
    unsigned int mCount;
private:
    IndexBuffer(const IndexBuffer& other);
    IndexBuffer& operator=(const IndexBuffer& other);
public:
    IndexBuffer();
    ~IndexBuffer();
    void Set(unsigned int* rr, unsigned int len);
    void Set(std::vector<unsigned int>& input);
    unsigned int Count();
    unsigned int GetHandle();
};
```

在本节中，你声明了一个新的`IndexBuffer`类。在下一节中，你将开始实现实际的索引缓冲区。

## 实现 IndexBuffer 类

索引缓冲区允许你使用索引几何体渲染模型。想象一个人体模型；网格中几乎所有的三角形都是相连的。这意味着许多三角形可能共享一个顶点。而不是存储每个单独的顶点，只存储唯一的顶点。索引到唯一顶点列表的缓冲区，即索引缓冲区，用于从唯一顶点创建三角形，如下所示：

1.  创建一个新文件，`IndexBuffer.cpp`。你将在这个文件中实现`IndexBuffer`类。构造函数需要生成一个新的 OpenGL 缓冲区，析构函数需要删除该缓冲区：

```cpp
    IndexBuffer::IndexBuffer() {
        glGenBuffers(1, &mHandle);
        mCount = 0;
    }
    IndexBuffer::~IndexBuffer() {
        glDeleteBuffers(1, &mHandle);
    }
    ```

1.  `IndexBuffer`对象内部的计数和 OpenGL 句柄的 getter 函数是微不足道的：

```cpp
    unsigned int IndexBuffer::Count() {
        return mCount;
    }
    unsigned int IndexBuffer::GetHandle() {
        return mHandle;
    }
    ```

1.  `IndexBuffer`类的`Set`函数需要绑定`GL_ELEMENT_ARRAY_BUFFER`。除此之外，逻辑与属性的逻辑相同：

```cpp
    void IndexBuffer::Set(unsigned int* inputArray, unsigned int arrayLengt) {
        mCount = arrayLengt;
        unsigned int size = sizeof(unsigned int);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mHandle);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, size * mCount, inputArray, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    void IndexBuffer::Set(std::vector<unsigned int>& input) {
        Set(&input[0], (unsigned int)input.size());
    }
    ```

在本节中，你围绕索引缓冲区构建了一个抽象。在下一节中，你将学习如何使用索引缓冲区和属性来渲染几何体。

# 渲染几何体

你已经有了处理顶点数据、统一和索引缓冲区的类，但没有任何代码来绘制它们。绘制将由四个全局函数处理。你将有两个`Draw`函数和两个`DrawInstanced`函数。你将能够使用或不使用索引缓冲区来绘制几何体。

创建一个新文件，`Draw.h`。你将在这个文件中实现`Draw`函数，如下所示：

1.  声明一个`enum`类，定义绘制时应该使用的基本图元。大多数情况下，你只需要线、点或三角形，但有些额外的类型可能也会有用：

```cpp
    enum class DrawMode {
        Points,
        LineStrip,
        LineLoop,
        Lines,
        Triangles,
        TriangleStrip,
        TriangleFan
    };
    ```

1.  接下来，声明`Draw`函数。`Draw`函数有两个重载——一个接受索引缓冲区和绘制模式，另一个接受顶点数量和绘制模式：

```cpp
    void Draw(IndexBuffer& inIndexBuffer, DrawMode mode);
    void Draw(unsigned int vertexCount, DrawMode mode);
    ```

1.  像`Draw`一样，声明两个`DrawInstanced`函数。这些函数具有类似的签名，但多了一个参数——`instanceCount`。这个`instanceCount`变量控制着几何体的实例数量将被渲染：

```cpp
    void DrawInstanced(IndexBuffer& inIndexBuffer, 
             DrawMode mode, unsigned int instanceCount);
    void DrawInstanced(unsigned int vertexCount, 
             DrawMode mode, unsigned int numInstances);
    ```

创建一个新文件，`Draw.cpp`。你将在这个文件中实现与绘制相关的功能，如下所示：

1.  你需要能够将`DrawMode`枚举转换为`GLenum`。我们将使用一个静态辅助函数来实现这一点。这个函数唯一需要做的事情就是弄清楚输入的绘制模式是什么，并返回适当的`GLenum`值：

```cpp
    static GLenum DrawModeToGLEnum(DrawMode input) {
        switch (input) {
            case DrawMode::Points: return  GL_POINTS;
            case DrawMode::LineStrip: return GL_LINE_STRIP;
            case DrawMode::LineLoop: return  GL_LINE_LOOP;
            case DrawMode::Lines: return  GL_LINES;
            case DrawMode::Triangles: return  GL_TRIANGLES;
            case DrawMode::TriangleStrip: 
                           return  GL_TRIANGLE_STRIP;
            case DrawMode::TriangleFan: 
                           return   GL_TRIANGLE_FAN;
        }
        cout << "DrawModeToGLEnum unreachable code hit\n";
        return 0;
    }
    ```

1.  接受顶点数的`Draw`和`DrawInstanced`函数很容易实现。`Draw`需要调用`glDrawArrays`，而`DrawInstanced`需要调用`glDrawArraysInstanced`：

```cpp
    void Draw(unsigned int vertexCount, DrawMode mode) {
        glDrawArrays(DrawModeToGLEnum(mode), 0, vertexCount);
    }
    void DrawInstanced(unsigned int vertexCount, 
         DrawMode mode, unsigned int numInstances) {
        glDrawArraysInstanced(DrawModeToGLEnum(mode), 
                              0, vertexCount, numInstances);
    }
    ```

1.  接受索引缓冲区的`Draw`和`Drawinstanced`函数需要将索引缓冲区绑定到`GL_ELEMENT_ARRAY_BUFFER`，然后调用`glDrawElements`和`glDrawElementsInstanced`：

```cpp
    void Draw(IndexBuffer& inIndexBuffer, DrawMode mode) {
        unsigned int handle = inIndexBuffer.GetHandle();
        unsigned int numIndices = inIndexBuffer.Count();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);
        glDrawElements(DrawModeToGLEnum(mode), 
                       numIndices, GL_UNSIGNED_INT, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    void DrawInstanced(IndexBuffer& inIndexBuffer, 
             DrawMode mode, unsigned int instanceCount) {
        unsigned int handle = inIndexBuffer.GetHandle();
        unsigned int numIndices = inIndexBuffer.Count();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle);
        glDrawElementsInstanced(DrawModeToGLEnum(mode),
            numIndices, GL_UNSIGNED_INT, 0, instanceCount);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    ```

到目前为止，您已经编写了加载着色器、创建和绑定 GPU 缓冲区以及将统一变量传递给着色器的代码。现在绘图代码也已实现，您可以开始显示几何图形了。

在下一节中，您将学习如何使用纹理使渲染的几何图形看起来更有趣。

# 使用纹理

本书中编写的所有着色器都假定正在渲染的漫反射颜色来自纹理。纹理将从`.png`文件加载。所有图像加载都将通过`stb_image`完成。

`Stb`是一组单文件公共领域库。我们只会使用图像加载器；您可以在 GitHub 上找到整个`stb`集合[`github.com/nothings/stb`](https://github.com/nothings/stb)。

## 添加 stb_image

您将使用`stb_image`加载纹理。您可以从[`github.com/nothings/stb/blob/master/stb_image.h`](https://github.com/nothings/stb/blob/master/stb_image.h)获取头文件的副本。将`stb_image.h`头文件添加到项目中。

创建一个新文件`stb_image.cpp`。这个文件只需要声明`stb_image`实现宏并包含头文件。它应该是这样的：

```cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
```

## 纹理类声明

创建一个新文件`Texture.h`。您将在这个文件中声明`Texture`类。`Texture`类只需要一些重要的函数。它需要能够从文件加载纹理，将纹理索引绑定到统一索引，并取消激活纹理索引。

除了核心函数之外，该类还应该有一个默认构造函数、一个方便的构造函数（接受文件路径）、一个析构函数和一个获取`Texture`类内包含的 OpenGL 句柄的 getter。复制构造函数和赋值运算符应该被禁用，以避免两个`Texture`类引用相同的 OpenGL 纹理句柄：

```cpp
class Texture {
protected:
    unsigned int mWidth;
    unsigned int mHeight;
    unsigned int mChannels;
    unsigned int mHandle;
private:
    Texture(const Texture& other);
    Texture& operator=(const Texture& other);
public:
    Texture();
    Texture(const char* path);
    ~Texture();
    void Load(const char* path);
    void Set(unsigned int uniform, unsigned int texIndex);
    void UnSet(unsigned int textureIndex);
    unsigned int GetHandle();
};
```

## 实现纹理类

创建一个新文件`Texture.cpp`。`Texture`类的定义将放在这个文件中。`Texture`类的默认构造函数需要将所有成员变量设置为`0`，然后生成一个 OpenGL 句柄。

`Load`函数可能是`Texture`类中最重要的函数；它负责加载图像文件。图像文件的实际解析将由`stbi_load`处理：

1.  方便的构造函数生成一个新的句柄，然后调用`Load`函数，该函数将初始化`Texture`类的其余成员变量，因为`Texture`类的每个实例都持有一个有效的纹理句柄：

```cpp
    Texture::Texture() {
        mWidth = 0;
        mHeight = 0;
        mChannels = 0;
        glGenTextures(1, &mHandle);
    }
    Texture::Texture(const char* path) {
        glGenTextures(1, &mHandle);
        Load(path);
    }
    Texture::~Texture() {
        glDeleteTextures(1, &mHandle);
    }
    ```

1.  `stbi_load`需要一个图像文件的路径以及图像的宽度、高度和通道数的引用。最后一个参数指定每个像素的组件数。通过将其设置为`4`，所有纹理都将以 RGBA 通道加载。接下来，使用`glTexImage2D`将纹理上传到 GPU，并使用`glGenerateMipmap`生成图像的适当 mipmap。将包装模式设置为重复：

```cpp
    void Texture::Load(const char* path) {
        glBindTexture(GL_TEXTURE_2D, mHandle);
        int width, height, channels;
        unsigned char* data = stbi_load(path, &width, 
                                        &height, 
                                        &channels, 4);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, 
           height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 
                        GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 
                        GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,
                        GL_NEAREST_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,
                        GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        mWidth = width;
        mHeight = height;
        mChannels = channels;
    }
    ```

1.  `Set`函数需要激活一个纹理单元，将`Texture`类包含的句柄绑定到该纹理单元，然后将指定的统一索引设置为当前绑定的纹理单元。`Unset`函数取消绑定指定纹理单元的当前纹理：

```cpp
    void Texture::Set(unsigned int uniformIndex, 
                      unsigned int textureIndex) {
        glActiveTexture(GL_TEXTURE0 + textureIndex);
        glBindTexture(GL_TEXTURE_2D, mHandle);
        glUniform1i(uniformIndex, textureIndex);
    }
    void Texture::UnSet(unsigned int textureIndex) {
        glActiveTexture(GL_TEXTURE0 + textureIndex);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
    }
    ```

1.  `GetHandle`获取函数很简单：

```cpp
    unsigned int Texture::GetHandle() {
        return mHandle;
    }
    ```

`Texture`类将始终使用相同的 mipmap 级别和包装参数加载纹理。对于本书中的示例，这应该足够了。您可能希望尝试为这些属性添加 getter 和 setter。

在下一节中，您将实现顶点和片段着色器程序，这是绘制所需的最后一步。

# 简单的着色器

渲染抽象已完成。在绘制任何东西之前，您需要编写着色器来指导绘制的方式。在本节中，您将编写一个顶点着色器和一个片段着色器。片段着色器将在本书的其余部分中使用，而本书后面部分使用的顶点着色器将是这里介绍的一个变体。

## 顶点着色器

顶点着色器负责将模型的每个顶点通过模型、视图和投影管道，并将任何所需的光照数据传递给片段着色器。创建一个新文件，`static.vert`。您将在这个文件中实现顶点着色器。

顶点着色器需要三个 uniform 变量——模型、视图和投影矩阵。这些 uniform 变量需要用来转换顶点。每个单独的顶点由三个属性组成——位置、法线和一些纹理坐标。

顶点着色器将三个变量输出到片段着色器中，即世界空间中的法线和片段位置，以及纹理坐标：

```cpp
#version 330 core
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
in vec3 position;
in vec3 normal;
in vec2 texCoord;
out vec3 norm;
out vec3 fragPos;
out vec2 uv;
void main() {
    gl_Position = projection * view * model * 
                  vec4(position, 1.0);

    fragPos = vec3(model * vec4(position, 1.0));
    norm = vec3(model * vec4(normal, 0.0f));
    uv = texCoord;
}
```

这是一个最小的顶点着色器；它只将顶点通过模型视图和投影管道。这个着色器可以用来显示静态几何图形或 CPU 蒙皮网格。在下一节中，您将实现一个片段着色器。

## 片段着色器

创建一个新文件，`lit.frag`。这个文件中的片段着色器将在本书的其余部分中使用。一些章节将介绍新的顶点着色器，但片段着色器始终保持不变。

片段着色器从纹理中获取对象的漫反射颜色，然后应用单向光。光照模型只是*N*点*L*。由于光没有环境项，模型的某些部分可能会呈现全黑：

```cpp
#version 330 core
in vec3 norm;
in vec3 fragPos;
in vec2 uv;
uniform vec3 light;
uniform sampler2D tex0;
out vec4 FragColor;
void main() {
    vec4 diffuseColor = texture(tex0, uv);
    vec3 n = normalize(norm);
    vec3 l = normalize(light);
    float diffuseIntensity = clamp(dot(n, l), 0, 1);
    FragColor = diffuseColor * diffuseIntensity;
}
```

重要信息：

想了解更多关于 OpenGL 中光照模型的知识？请访问[`learnopengl.com/Lighting/Basic-Lighting`](https://learnopengl.com/Lighting/Basic-Lighting)。

这是一个简单的片段着色器；漫反射颜色是通过对纹理进行采样获得的，强度是一个简单的定向光。

# 总结

在本章中，您学会了如何在 OpenGL API 的顶层编写一个抽象层。在本书的大部分时间里，您将使用这些类来绘制东西，但是一些零散的 OpenGL 调用可能会在我们的代码中找到它们的位置。

以这种方式抽象化 OpenGL 将让未来的章节专注于动画，而不必担心底层 API。将这个 API 移植到其他后端也应该很简单。

本章有两个示例——`Chapter06/Sample00`，这是到目前为止使用的代码，以及`Chapter06/Sample01`，显示一个简单的纹理和光照平面在原地旋转。`Sample01`是如何使用到目前为止编写的代码的一个很好的例子。

`Sample01`还包括一个实用类`DebugDraw`，本书不会涉及。该类位于`DebugDraw.h`和`DebugDraw.cpp`中。`DebugDraw`类可以用于快速绘制调试线，具有简单的 API。`DebugDraw`类效率不高；它只用于调试目的。

在下一章中，您将开始探索 glTF 文件格式。glTF 是一种可以存储网格和动画数据的标准格式。这是本书其余部分将使用的格式。
