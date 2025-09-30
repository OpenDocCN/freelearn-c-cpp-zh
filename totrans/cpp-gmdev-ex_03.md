# Mathematics and Graphics Concepts

Before we begin rendering objects, it is essential that you are familiar with the math that will be used for the projects in this book. Mathematics plays a crucial role in game development, and graphics programming generally uses vectors and matrices extensively. In this chapter, you will understand where these math concepts can come in handy. First, we'll go over some key mathematical concepts and then apply them so that we can work with space transformations and render pipelines. There are dedicated books that cover all the math-related topics that you'll need for game development. However, since we will be covering graphics programming with C++, other mathematics topics are out of the scope of this book.

In the upcoming chapters, we will be using the OpenGL and Vulkan graphics APIs to render our objects and use the GLM math library to do the maths. In this chapter, we will explore the process of creating a 3D object in a virtual world using matrix and vector transforms. Then, we will look at how we can transform a 3D point into a 2D location using space transforms, as well as how the graphics pipeline helps us achieve this.

In this chapter, we will cover the following topics:

*   3D coordinate systems
*   Vectors
*   Matrices
*   GLM OpenGL mathematics
*   OpenGL data types
*   Space transformations
*   Render pipeline

# 3D coordinate systems

Before we can specify a location, we have to specify a coordinate system. A 3D coordinate system has three axes: the *x *axis, the *y* axis, and the *z* axis. These three axes start from the origin of where the three axes intersect.

The positive x axis starts from the origin and starts moving endlessly in a certain direction, while the negative *x* axis moves in the opposite direction. The positive *y *axis starts from the origin and moves in an upward direction at 90 degrees to the *x *axis, and the negative *y *axis moves in the opposite direction. This describes a 2D XY plane, which forms the basis of a 2D coordinate system.

The positive *z *axis starts from the same origin as the *x* and *y* axes and is perpendicular to the X and Y axes. The positive *z* axis can go in either direction of the XY plane in order to form a 3D coordinate system.

Assuming that the positive *x* axis is going to the right and the positive *y* axis is going up, then the *z *axis can either go into or out of the screen. This is because the *z *axis is perpendicular to the *x* and *y* axes.

When the positive *z *axis moves into the screen, this is known as a **left-handed coordinate system**. When the positive *z *axis comes out of the screen, this is known as a **ight-handed coordinate system**.

Extend your right arm so that it's in front of you, with your palm facing toward you, and make a fist. Extend your thumb to the right, and then extend your index finger upward. Now, extend your middle finger so that it faces you. This can be used to explain the right-handed coordinate system.

The thumb represents the direction of the positive *x* axis, the index finger represents the direction of the positive *y* axis, and the middle finger is the direction of the positive *z *axis. OpenGL, Vulkan, or any other graphics framework that uses these axes also use this coordinate system.

For the left-handed coordinate system, extend your left arm out so that it's in front of you, with the palm of your hand facing away from you, and make a fist. Next, extend your thumb and index finger in the right and upward directions, respectively. Now, extend your middle finger so that it's away from you. In this case, the thumb also represents the direction of the *x* axis and the index finger is pointing in the direction of the positive *y* axis. The *z *axis (the middle finger) is now facing away from you. Direct3D of DirectX uses this coordinate system.

In this book, since we are going to be covering OpenGL and Vulkan, we will be using the **right-handed coordinate system**:

![](img/173e45a5-bdff-42fc-9d6b-ae4e42108aa6.png)

# Points

Now that we have defined the coordinate system, we can specify what a point is. A 3D point is a location in 3D space that's specified by distance in terms of the **X**, **Y**, and **Z** axes and the origin of the coordinate system. It is specified as (X, Y, Z) where X, Y, and Z are the distance from the origin. But what is this origin we speak of? The origin is also the point where the three axes meet. The origin is at (0, 0, 0), and the location of the origin is specified in the coordinate system, as shown in the following diagram:

![](img/7c1608c8-03a6-4a61-84f8-75278ce57907.png)

To specify points within a coordinate system, imagine that, in each direction, the axis is made of a smaller unit. This unit could be 1 millimeter, 1 centimeter, or 1 kilometer, depending on how much data you have.

If we just look at the X and Y axes, this would look something like this:

![](img/14788c58-1d46-42dc-82c7-1470797f8f59.png)

If we look at the *x* axis, the values 1 and 2 specify the distance along the axis of that point from the origin, which is at value 0\. So, point 1 in the *x* axis is at (1, 0, 0) along the *x* axis. Similarly, point 1, which is along the *y* axis, is at (0, 1, 0).

In addition, the location of the red dot will be at (1, 1, 0); that is, 1 unit along the *x* axis and 1 unit along the *y* axis. Since Z is 0, we specify that its value is 0.

Similarly, the points in 3D space are represented as follows:

![](img/96c28731-e7cc-4c6c-81a1-ee8e7285d42c.png)

# Vectors

A vector is a quantity that has a magnitude and a direction. Examples of quantities that have a magnitude and direction are displacement, velocity, acceleration, and force. With displacement, you can specify the direction as well as the net distance that the object has moved by.

The difference between speed and velocity is that speed only specifies the speed that an object is moving at, but doesn't establish the direction the object is moving in. However, velocity specifies the magnitude, which includes speed and direction. Similar to velocity, we have acceleration. A form of acceleration is gravity, and we know that this always acts downward and is always approximately 9.81 m/s²  – well, at least on Earth. It is 1/6^(th ) of this on the moon.

An example of force is weight. Weight also acts downward and is calculated as mass multiplied by acceleration.

Vectors are graphically represented by a pointed line segment, with the length of the line denoting the magnitude of the vector and the pointed arrow denoting the direction of the vector. We can move around a vector since doing this doesn't change the magnitude or direction of it.

Two vectors are said to be equal if they both have the same magnitude and direction, even if they are in different locations. Vectors are denoted by arrow marks above the letter.

In the following diagram, vectors ![](img/2a1e581e-d673-4ab8-a190-742cc0f7cbfd.png) and ![](img/8bc6892d-8e78-4fcc-ba73-e642288cde44.png) start in different locations. Since the direction and magnitude of the are arrows the same, they are equal:

![](img/fa8b7924-63fc-4bfc-9e1a-140d1b32793a.png)

In a 3D coordinate system, a vector is specified by the coordinates with respect to the coordinate system. In the following diagram, the vector, ![](img/ac329482-d883-4a99-9f36-a6f0d7f49832.png) is equal to (2, 3, 1) and is denoted as ![](img/1798f038-6515-471c-8a07-2b61201db57d.png) =![](img/2dae2535-79ec-444f-b2eb-59341ded4914.png):

![](img/ffc09b70-20e0-4a4f-a06e-67a53498412f.png)

# Vector operations

Just like scalar information, vectors can also be added, subtracted, and multiplied. Suppose you have two vectors, ![](img/214bb708-9019-41ac-b71a-72773b5506e5.png) and ![](img/c708a57d-c191-4fa0-bc41-daea6143fb64.png) , where ![](img/343aaa7e-9caa-4dfe-8520-6d6788753578.png)= (a[x], a[y], a[z]) and  ![](img/8ac3e72f-eca2-453e-8494-35b70b081f0d.png)= (b[x], b[y], b[z]). Let's look at how we can add and subtract these vectors to/from each other.

When adding vectors, we add the components individually to create a new vector:

![](img/9d1bce52-2dfd-4ec3-9d3a-e74b6cef8c99.png)= ![](img/0dbf3fce-6ebc-4977-a9d4-aa03a1d86851.png) +  ![](img/09c33362-d2f1-4f5a-b394-ea0bc3dacc7e.png)

![](img/c74fa717-c4a9-4e66-99d4-d5aa515e3c2e.png)= ((ax + bx) , (ay + by) , (az + bz ))

Now, let's visualize the addition of two vectors in a graph. The Z value is kept as 0.0 for convenience. Here, ![](img/973328f0-7acb-424f-8ae3-e391aad4e811.png)=(1.0, 0.4, 0.0 ) and  ![](img/4ea4a919-bf8d-46ea-9143-c0f4cf3c5abc.png)=(0.6, 2.0, 0.0), which means that the resultant vector,  ![](img/df084c7d-cb1b-42ab-9c8f-d0628d213d90.png)= ![](img/c7261b2a-0c1e-428d-9a36-fc659f099f4e.png) +  ![](img/01c1a38c-3785-497f-8ce2-86647d472448.png), = (1.0 + 0.6 , 0.4 + 2.0, 0.0 + 0.0) = (1.6, 2.4 , 0.0):

![](img/521124c0-58e2-4239-bcb3-f3a0cd0943fb.png)

Vectors are also commutative, meaning that  ![](img/695a78c8-3cbe-46b4-b06f-148bc930aa88.png) +  ![](img/341c8b1d-aece-41ec-93a7-c5aa1eecc1e4.png)  will give the same result as ![](img/f5079cea-ebc3-4e26-8636-23f403cfd5e9.png) + ![](img/eb876de9-01a5-4b7f-8b67-0ec8161f244c.png).

 However, if we add ![](img/34defdb7-77b3-4f41-b8f4-1b336a81016a.png) to ![](img/4b98cad1-6b37-4b07-90bc-cfed48e4b97b.png), then the dotted line will go from the vector ![](img/341c8b1d-aece-41ec-93a7-c5aa1eecc1e4.png) to vector ![](img/041962c2-03d2-43ad-8df1-89ce6c08461d.png) as shown in the preceding figure.  Furthermore, in vector subtraction, we subtract the individual components of the vectors to create a new vector:

![](img/6161b965-e04f-4219-96bb-1547d2383e8c.png)= ![](img/400bff09-6950-4856-9178-dda2eae4243a.png) -  ![](img/d289cba8-c3bc-4135-a3ab-9bc7227d8028.png)

![](img/041962c2-03d2-43ad-8df1-89ce6c08461d.png)= ((ax - bx) , (ay - by) , (az - bz ))

Now, let's visualize the subtraction of two vectors in a graph.

Here,  ![](img/baafd86e-4698-4ba9-8552-53869a81e01a.png) = (1.0, 0.4, 0.0 ) and  ![](img/eba72c42-6fff-4904-ae61-0a3dff1662a4.png)= (0.6, 2.0, 0.0). Therefore, the resultant vector,  ![](img/e5218906-5593-4155-bb71-5d0d573b9545.png)= ![](img/9cb94812-72b8-40c5-b0fb-12eb9639e4f6.png) -  ![](img/ec2af80a-391a-42d6-825b-c5e413685224.png), = (1.0 - 0.6, 0.4 - 2.0, 0.0 - 0.0) = (0.4, -1.6, 0.0):

![](img/8f597d24-de01-4226-8ca0-aa502bb06312.png)

If vectors A and B are equal, the result will be a zero vector with all three components as zero.

If  ![](img/fc0f6b71-42a0-4fe6-9a3e-1cd3e8ed70b7.png)= ![](img/356fd3ab-9b48-4c24-82ce-61a857a6dad3.png), this means that a[x] = b[x] , a[y] = b[y] , a[z] = b[z]. If this is the case, then, ![](img/fb484272-bc03-417f-ab5b-58e9eba98f9e.png)= ![](img/4728f4f8-ce7e-4d20-9503-27b0ad8e2441.png) -  ![](img/73224def-1249-491f-ad09-f32f5a84dafb.png)= (0, 0, 0).

We can multiply a scalar by a vector. The result is a vector with each component of the vector being multiplied by the scalar.

For example, if A is multiplied by a single value of s, we will have the following result:

![](img/83a4ffeb-5a2c-489a-92df-0bb227e997e6.png)

# Vector magnitude

The magnitude of the vector is equal to the length of the vector itself. But how do we calculate it mathematically?

The magnitude of a vector is given by the Pythagorean theorem, which specifies that, in a right-handed triangle, the square of the length of a diagonal is equal to the sum of the squares of the adjacent sides. Let's take a look at the following right-handed triangle, *c² = x² + y²*:

![](img/841cc4e8-7755-4a4b-af66-f282e0a387b4.png)

This can be extended to three dimensions with *c² = x² + y² + z²*.

Magnitudes of vectors are indicated by double vertical bars, so the magnitude of a vector, 
![](img/6be0b0cc-8c0a-4aeb-8e23-e0568ec4947f.png) is denoted by ![](img/9e90e49e-a576-416f-9463-986d8b3cc5cb.png). The magnitude is always greater than or equal to zero.

So, if vector A = (X, Y, Z), then the magnitude is given by the following equation: 

![](img/228d2181-2e99-4e79-a366-39c23826586e.png)

If ![](img/20878924-54f2-4e0b-8d4f-375b4a6681d6.png) = (3, -5, 7), then we get the following:

![](img/26935f31-c45e-4df4-beb5-5ead26087101.png)

Therefore,  ![](img/d5f5d2ef-1ae8-4fb5-9035-375298c51c41.png) is 9.11 units long.

# Unit vectors

In some cases, we don't care about the magnitude of the vector; we just want to know the direction of the vector. To find this out, we want the length of the vector in the X, Y, and Z directions to be equal to 1\. Such vectors are called unit vectors or normalized vectors.

In unit vectors, the X, Y, and Z components of the vector are divided by the magnitude to create a vector of unit length. They are denoted by a hat on top of the vector name instead of an arrow. So, a unit vector of A will be denoted as ![](img/dc50564b-a440-479d-b8eb-5a3b18bd05dd.png), like so:

![](img/daed6188-72d0-433c-bffa-b594f0975b6c.png)

When a vector is converted into a unit vector, it is said to be normalized. This means that the value is always between 0.0 and 1.0\. The original value has been rescaled to be in this range. Let's normalize the vector ![](img/511f231c-a556-4ee7-aab1-f5c0dbcf3baa.png)= (3, -5, 7).

First, we have to calculate the magnitude of ![](img/bf8003d7-bcd0-4ec4-b375-6951969677c3.png), which we have already done (it's 9.11).

So, the unit vector  is as follows:

![](img/eb11fae8-5f76-49c4-bbb5-99e1e509f170.png)

# The dot product

The dot product is a type of vector multiplication in which the resultant vector is a scalar. It is also referred to as a scalar product for the same reason. The scalar product of two vectors is the sum of the products of the corresponding components.

If you have two vectors, A = (a[x], a[y], a[z]) and B = (b[x], b[y], b[z]), this is given by the following equation:

![](img/39934380-70d0-4e52-9201-8a748093c538.png)

The dot product of two vectors is also equal to the cosine of the angle between the vectors that have been multiplied by the magnitudes of both vectors. Note that the dot product is represented as a dot between the vector:

![](img/ac795e9e-2f01-4909-9dd8-5191826ca2cc.png)

θ is always between 0 and π. By putting an equation of 1 and 2 together, we can figure out the angle between two vectors:

![](img/7c319066-651c-490f-9d71-5d34200fd421.png)

  Consequently we get: 

![](img/883e2f82-bc67-4c79-953b-42e31ff5a246.png)

This form has some unique geometric properties:

*   If ![](img/d9e77aee-35ae-4a77-9456-8d3befb50196.png)= 0, then ![](img/821d50ac-4d69-4174-82d6-fbce7ae8541e.png) is perpendicular to ![](img/a9c31b6b-9b19-4fdd-9775-9cbe74271628.png), that is, cos 90 = 0.
*   If ![](img/cb663a1e-8ba7-4eeb-89ea-551bf1b18b63.png) = ![](img/3a2bb232-fd3a-41b6-8572-e1a2d268d900.png) , then the two vectors are parallel to each other, that is, cos 0 = 1.
*   If ![](img/bb285625-eec7-47da-886b-69606c36723f.png)> 0, then the angle between the vectors is less than 90 degrees.
*   If ![](img/524fcaca-2c97-45f3-ad1f-c50929ac458e.png)< 0, then the angle between the vectors is greater than 90 degrees.

Now, let's look at an example of a dot product.

If ![](img/20486fd6-d306-47bf-8394-5512d2231b56.png)= (3, -5, 7) and  ![](img/f2c9b707-1e49-4113-adf6-95c35928682c.png) = (2, 4 , 1), then ![](img/274f44c4-038e-47f0-a41c-26251feb4a15.png) = 9.110 and ![](img/2fae8534-ebaf-4785-b2c6-f30c784a6d73.png).

Next, we calculate like so:

![](img/12ad5de1-66f3-4449-a5b8-89ad506cdfd4.png)

# The cross product

The c oss product is another form of vector multiplication in which the resultant product of the multiplication is another vector. Taking the cross product between ![](img/da7b1b6f-4c7e-493b-b661-d60c12afcdc2.png) and ![](img/b1027b57-1635-4312-9d74-76cf0115b7d3.png) will result in a third vector that is perpendicular to vectors ![](img/2a67360a-658e-4526-9ff3-7d77546eece8.png) and ![](img/f521ea0d-f08d-4e19-b341-d867fe50082a.png) .

If you have two vectors,  ![](img/1fb5a9f4-3b10-4592-8eeb-827a24111d48.png) = (a[x], a[y], a[z]) and  ![](img/aa92a4fd-2f9e-41a2-a4f7-4528edb9bc84.png) = (b[x], b[y], b[z]), then ![](img/671707f6-6e01-433e-81d7-a368278356c9.png) is given as follows:

![](img/638d5253-1053-409a-91c1-228c6761e795.png)

The following are matrix and graphical implementations of the cross product between vectors:

![](img/199f16fe-6475-4728-9686-b5741fdcaefe.png) ![](img/fdb31d82-e37f-463b-aef8-4a384926c543.png)

The direction of the resultant normal vector will obey the right-hand rule, where curling the fingers on your right hand (from ![](img/0f2bdafb-8c63-48e7-86fa-d27f8d18dff1.png) to ![](img/70a3b724-bf19-4392-b0df-61d7f52ae352.png)) will cause the thumb to point in the direction of the resultant normal vector (![](img/20ec9163-4b0e-4971-97f9-48d3a7e0f5c6.png)).

Also, note that the order in which you multiply the vectors is important because if you multiply the other way around, then the resultant vector will point in the opposite direction.

The cross product is very useful when we want to find the normal of the face of a polygon, such as a triangle.

The following equation helps us find the cross product of vectors  ![](img/3c5a82d4-e513-4969-ad08-16a5c8233559.png)= (3, -5, 7) and   ![](img/749b7e8e-3a7d-4e37-82ec-924263d4bebd.png)= (2, 4 , 1):

C = A × B = (ay bz - az by, az bx - ax bz, , ax by - ay bx)

= (-5 * 1 - 7*4 , 7 * 2 - 3 * 1, 3 * 4 - (-5) * 2 )

= (-5-28, 14 - 3, 12 + 10)

= (-33, 11, 22)

# Matrices

In computer graphics, matrices are used to calculate object transforms such as translation, that is, movement, scaling in the X, Y, and Z axes, and rotation around the X, Y, and Z axes. We will also be changing the position of objects from one coordinate system to an other, which is known as space transforms. We will see how matrices work and how they help to simplify the mathematics we have to use.

Matrices have rows and columns. A matrix with *m* number of rows and *n* number of columns is said to be a matrix of size *m × n*. Each element of a matrix is represented as indices *ij*, where *i* specifies the row number and *j* represents the column number.

So, a matrix, M, of size 3 × 2 is represented as follows:

![](img/f1a7ed5a-9ef6-475f-b376-c4d245fd7d73.png)

Here, matrix M has three rows and two columns and each element is represented as m11, m12, and so on until m32, which is the size of the matrix.

In 3D graphics programming, we mostly deal with 4 × 4 matrices. Let's look at another matrix that's 4 x 4 in size:

![](img/a86f16b4-e714-4ee2-86ba-b7047c8cc6b4.png)

Matrix A can be presented as follows:

![](img/e618f69a-a18c-4d52-873e-447fc3567f1c.png)

Here, the elements are A[11] = 3, A[32] = 1, and A[44] = 1 and the dimension of the matrix is 4 × 4.

We can also have a single-dimension matrix with a vector shown as follows. Here, B is called the row vector and C is called a column vector:

![](img/3e0e45e4-a37f-45c2-8fda-43e0a918ba04.png)

*   Two matrices are equal if the number of rows and columns are the same and if the corresponding elements are of the same value.
*   Two matrices can be added together if they have the same number of rows and columns. We add each element of the corresponding location to both matrices to get a third matrix that has the same dimensions as the added matrices.

# Matrix addition and subtraction

Consider the following two matrices, A and B. Both of these are 3 x 3 in size:

![](img/9d0d2993-e54b-4811-9527-0645a5f98517.png)

Here, if C = A + B, then this can be represented like so:

![](img/4ac3dcf8-3ea9-4e29-87a9-c2d84aaa7cc1.png)

Matrix subtraction works in the same way when each element of the matrix is subtracted from the corresponding element of the other matrix.

# Matrix multiplication

Let's look at how a scalar value can be multiplied to a matrix. We can do this by multiplying each element of the matrix by the same scalar value. This will give us a new matrix that has the same dimensions as the original matrix.

Again, consider a Matrix, A, that's been multiplied by some scalars. Here, s×A, as follows:

![](img/01a79209-4c6f-43df-a073-e0d25e0465f4.png)

Two A and B matrices can be multiplied together, provided that the number of columns in A is equal to the number of rows in B. So, if matrix A has the dimension a × b and B has the dimension X × Y, then for A to be multiplied by B, b should be equal to X:

The resultant size of the matrix will be a × Y. Two matrices can be multiplied together like so:

![](img/e6838ac1-f95d-4299-a2d4-b314e80bdd1f.png)

Here, the size of A is 3 × 2 and the size of B is 2 × 3, which means that the resultant matrix, C, will be 3 × 3 in size:

![](img/dd17ad54-a2df-42aa-a29f-93f54b0083e0.png)

However, keep in mind that matrix multiplication is not commutative, meaning that A×B ≠ B×A. In fact, in some cases, it isn't even possible to multiply the other way around, just like it isn't in this case. Here, we can't even multiply B×A since the number of columns of B is not equal to the number of rows of A. In other words, the internal dimensions of the matrices should match so that the dimensions are in the form of [a![](img/704dcf14-ac4f-4ac5-88d6-6e4892b833c2.png)t] and [t![](img/ce4238d9-5e4c-40a0-93f7-041fb1b9038c.png)b].

You can also multiply a vector matrix with a regular matrix, as follows:

![](img/82ccb486-b8b8-43a1-bfc5-f603a8e8f641.png)

The result will be a one-dimensional vector of size 3 × 1, as follows:

![](img/d892153f-cd44-4ae0-bf49-8cf46e8dd5fb.png)

Note that when multiplying the matrix with the vector-matrix, the vector is to the right of the matrix. This is done so that the matrix of size 3 × 3 is able to multiply the vector-matrix of size 3 × 1.

When we have a matrix with just one column, this is called a column-major matrix. So, matrix C is a column-major matrix, just like matrix V.

If the same vector, V, was expressed with just a row, it would be called a row-major matrix. This can be represented like so:

![](img/a3184621-c6d0-4e20-8a54-8cb49544296d.png)

So, how would we multiply a matrix, A, of size 3 × 3 with a row-major matrix, V, of size 1 × 3 if the internal dimensions don't match?

The simple solution here is, instead of multiplying matrix A × V, we multiply V × A. This way, the internal dimensions of the vector-matrix and the regular matrix will match 1 × 3 and 3 × 3, and the resultant matrix will also be a row-major matrix.

Throughout this book, we will be using the column-major matrix.

If we were going to use 4 × 4 matrices, for example, how would we multiply a 4 × 4 matrix using the coordinates of x, y, and z? 

When multiplying a 4 × 4 matrix with points X, Y, and Z, we add one more row to the column-major matrix and set the value of it to 1\. The new point will be (X, Y, Z, 1), which is called a homogeneous point. This makes it easy to multiply a 4 × 4 matrix with a 4 × 1 vector:

![](img/c19553c6-60de-4b74-9bc7-ad63331666fe.png)

Matrix multiplication can be extrapolated to the multiplication of one 4 × 4 matrix with another 4 × 4 matrix. Let's look at how we can do this:

![](img/b4576903-6bad-4c0e-ae57-ee41d4735e70.png)

![](img/97b1de9b-b9f2-474e-aa2d-3acab4df20c0.png)

# Identity matrix

An identity matrix is a special kind of matrix in which the number of rows is equal to the number of columns. This is known as a square matrix. In an identity matrix, the elements in the diagonal of the matrix are all 1, while the rest of the elements are 0.

Here is an example of a 4 × 4 identity matrix:

![](img/1b1f9d8f-0a4b-4439-9c65-3dbf5bf11cc8.png)

Identity matrices work similarly to how we get a result when we multiply any number with 1 and get the same number. Similarly, when we multiply any matrix with an identity matrix, we get the same matrix.

For example, A×I = A, where A is a 4 ×4 matrix and I is an identity matrix of the same size. Let's look at an example of this:

![](img/2d2257c9-ad98-4e5d-abe9-23183926bf08.png)

# Matrix transpose

A matrix transpose occurs when the rows and columns are interchanged with each other. So, the transpose of an m X n matrix is n X m. The transpose of any matrix, M, is written as M^T. The transpose of a matrix is as follows: 

![](img/8273987f-a36a-46c0-8005-b0fa07a0cc18.png)

Observe how the elements in the diagonal of the matrix remain in the same place but all the elements around the diagonal have been swapped.

In matrices, this diagonal of the matrix, which runs from the top left to the bottom right, is called the main diagonal.

Obviously, if you transpose a transposed matrix, you get the original matrix, so (A^T)^T = A.

# Matrix inverse

The inverse of any matrix is where any matrix, when multiplied by its inverse, will result in an identity matrix. For matrix M, the inverse of the matrix is denoted as M^(-1).

The inverse is very useful in graphics programming when we want to undo the multiplication of a matrix.

For example, Matrix M is equal to A × B × C × D, where A, B, C and D are also matrices. Now, let's say we want to know what A× B × C is instead of multiplying the three matrices, which is a two-step operation: first, you will multiply A with B and then multiply the resulting matrix with C. You can multiply M with D^(-1) as that will yield the same result:

![](img/62bbe4bc-41a7-491b-9648-3e2a4ffe67cf.png)

# GLM OpenGL mathematics

To carry out the mathematical operations we've just looked at in OpenGL and Vulkan projects, we will be using a header-only C++ mathematics library called GLM. This was initially developed to be used with OpenGL, but it can be used with Vulkan as well:

![](img/564a8c99-efc2-4116-b27b-f892e1a0b4c3.png)

The latest version of GLM can be downloaded from [https://glm.g-truc.net/0.9.9/index.html](https://glm.g-truc.net/0.9.9/index.html).

Apart from being able to create points and perform vector addition and subtraction, GLM can also define matrices, carry out matrix transforms, generate random numbers, and generate noise. The following are a few examples of how these functions can be carried out:

*   To define 2D and 3D points, we will need to include `#include <glm/glm.hpp>`, which uses the `glm` namespace. To define a 2D point in space, we use the following code:

```cpp
glm::vec2 p1 = glm::vec2(2.0f, 10.0f); 

Where the 2 arguments passed in are the x and y position. 
```

*   To define a 3D point, we use the following code:

```cpp
glm::vec3 p2 = glm::vec3(10.0f, 5.0f, 2.0f);   
```

*   A 4 x 4 matrix can also be created using `glm`, as shown in the following code. A 4 x 4 matrix is of the `mat4` type and can be created like this:

```cpp
glm::mat4 matrix = glm::mat4(1.0f); 

Here the 1.0f parameter passed in shows that the matrix is initialized as a identity matrix. 

```

*   For translation and rotation, you need to include the necessary GLM extensions, as shown in the following code:

```cpp
#include <glm/ext.hpp> 
glm::mat4 translation = glm::translate(glm::mat4(1.0f),  
glm::vec3(3.0f,4.0f, 8.0f)); 
```

*   To translate the object so that it's (`3.0`, `4.0`, `8.0`) from its current position, do the following:

```cpp
glm:: mat4 scale = glm::scale(glm::mat4(1.0f),  
glm::vec3( 2.0f, 2.0f, 2.0f));
```

*   We can also scale the value so that it's double its size in the *x*, *y*, and *z* directions:

```cpp
glm::mat4 rxMatrix = glm::rotate(glm::mat4(), glm::radians(45.0f), glm::vec3(1.0, 0.0, 0.0)); 
glm::mat4 ryMatrix = glm::rotate(glm::mat4(), glm::radians(25.0f), glm::vec3(0.0, 1.0, 0.0)); 
glm::mat4 rzMatrix = glm::rotate(glm::mat4(), glm::radians(10.0f), glm::vec3(0.0, 0.0, 1.0)); 
```

The preceding code rotates the object by `45,0f` degrees along the *x* axis, `25.0f` degrees along the *y* axis, and `10.0f` degrees along the *z *axis.

Note that we use `glm::radians()` here. This `glm` function converts degrees into radians. More GLM functions will be introduced throughout this chapter.

# OpenGL data types

OpenGL also has its own data types. These are portable across platforms.

OpenGL data types are prefixed with GL, followed by the data type. Consequently, a GL equivalent to an `int` variable is GLint, and so on. The following table shows a list of GL data types (the list can be viewed at [https://www.khronos.org/opengl/wiki/OpenGL_Type](https://www.khronos.org/opengl/wiki/OpenGL_Type)):

![](img/2ee7f1f6-fd88-4c9b-ae22-6d5cc8091ef6.png)

# Space transformations

The major job of 3D graphics is to simulate a 3D world and project that world into a 2D location, which is the viewport window. 3D or 2D objects that we want to render are nothing but collections of vertices. These vertices are then made into a collection of triangles to form the shape of the object sphere:

![](img/cdd49492-69b0-4cca-9a57-4a1428d38a1b.png)

The screenshot on the left shows the vertices that were passed in, whereas the screenshot on the right shows that the vertices were used to create triangles. Each triangle forms a small piece of the surface for the final shape of the object.

# Local/object space

When setting the vertices for any object, we start at the origin of a coordinate system. These vertices or points are placed and then connected to create the shape of the object, such as a triangle. This coordinate system that a model is created around is called the object space, model space, or local space:

![](img/4ac3a96b-8985-4734-bf4c-10ab0d6bf76b.png)

# World space

Now that we have specified the shape of the model, we want to place it in a scene, along with a couple of other shapes, such as a sphere and a cube. The cube and sphere shapes are also created using their own model space. When placing these objects into a 3D scene, we do so with respect to the coordinate system that the 3D objects will be placed in. This new coordinate system is called the world coordinate system, or world space.

Moving the object from the object space to the world space is done through matrix transforms. The local position of the object is multiplied by the world space matrix. Consequently, each vertex is multiplied by the world space matrix to transform its scale, rotation, and position from the local space to the world space.

The world space matrix is a product of scale, rotation, and translation matrices, as shown in the following formula:

*World Matrix = W = T × R ×S*

 S, R, and T are the scale, rotation, and translation of the local space relative to the world space, respectively. Let's take a look at each of them individually:

*   The scale matrix for a 3D space is a 4 x 4 matrix whose diagonal represents the scale in the *x*, *y*, and *z* directions, as follows:

![](img/d37b3b68-5f94-459f-b441-b5a6919bcb07.png)

*   The rotation matrix can take three forms, depending on which axis you are rotating the object on. *Rx*, *Ry*, and *Rz* are the matrices we use for rotation along each axis, as shown in the following matrix:

![](img/57fc346b-4394-4ef6-9e89-4860ef7d1fd3.png)![](img/2499f72f-7177-4a8f-ac49-b079846dc449.png)![](img/f47b6148-236c-4034-9c0a-a83a707ab072.png)

*   The translation matrix is an identity matrix where the last column represents the translation in the *x*, *y*, and *z* directions:

![](img/a36a7f29-dedf-44da-8d53-db1ab0cfea6f.png)

Now, we can get the world position by multiplying the local position of the objects with the world matrix, as follows:

Position[World] = Matrix[World] X Position[local]

# View space

For us to view the whole scene, we will need a camera. This camera will also decide which objects will be visible to us and which objects won't be rendered to the screen.

Consequently, we can place a virtual camera into the scene at a certain world location, as shown in the following diagram:

![](img/3c0e36de-a1eb-4fad-86c6-3c64682d14a4.png)

The objects in the scene are then transformed from the world space into a new coordinate system that's present at the location of the camera. This new coordinate system, which is at the location of the camera, is called the view space, camera space, or eye space. The *x *axis is red, the *y *axis is green, and the positive *z *axis is blue.

To transform the points from the world space to the camera space, we have to translate them using the negative of the virtual camera location and rotate them using the negative of the camera orientation.

However, there is an easier way to create the view matrix using GLM. We have to provide three variables to define the camera position, camera target position, and camera up vector, respectively:

```cpp
   glm::vec3cameraPos = glm::vec3(0.0f, 0.0f, 200.0f); 
   glm::vec3cameraFront = glm::vec3(0.0f, 0.0f, 0.0f); 
   glm::vec3cameraUp = glm::vec3(0.0f, 1.0f, 0.0f); 
```

We can use these variables to create a view matrix by calling the `lookAt` function and passing the camera position, the look at position, and up vector, as follows:

```cpp
   glm::mat4 viewMatrix = glm::lookAt(cameraPos, cameraPos + 
   cameraFront, cameraUp); 
```

Once we have the view matrix, the local positions can be multiplied by the world and the view matrix to get the position in the view space, as follows:

Position[view] = View[matrix] × World[matrix] × Position[local]

# Projection space

The next task is to project the 3D objects that can be viewed by the camera onto the 2D plane. Projection needs to be done in such a way that the furthest object appears smaller and the objects that are closer appear bigger. Basically, when viewing an object, the points need to converge at a vanishing point.

In the following screenshot, the image on the right shows a cube being rendered. Note how the lines on the longer sides are actually parallel.

However, when the same box is viewed from the camera, the same side lines converge, and when these lines are extended, they will converge at a point behind the box:

![](img/e549a806-59e4-4d19-858c-345497be1435.png)

Now, we will introduce one more matrix, called the projection matrix, which allows objects to be rendered with perspective projection. The vertices of the objects will be transformed using what is called a projection matrix to perform the perspective projection transformation.

In the projection matrix, we define a projection volume called the frustum. All the objects inside the frustum will be projected onto the 2D display. The objects outside the projection plane will not be rendered:

![](img/d29e913b-c2d3-46df-b413-9de5bc628a3a.png)

The projection matrix is created as follows:

![](img/e03b54bb-80f1-4690-87aa-c2fef3a73a77.png)

*q = 1/tan(FieldOfView/2)*

*A = q/Aspect Ratio*

*B = (zNear + zFar)/(zNear - zFar)*

*C = 2 *(zNear * zFar)/(zNear - zFar)*

The aspect ratio is the width of the projection plane divided by the height of the projection plane. *zNear* is the distance from the camera to the near plane. *zFar* is the distance from the camera to the far plane. The **field of view** (**FOV**) is the angle between the top and bottom planes of the view frustum.

In GLM, there is a function we can use to create the perspective projection matrix, as follows:

```cpp
GLfloat FOV = 45.0f; 
GLfloat width = 1280.0f; 
GLfloat height = 720.0f; 
GLfloat nearPlane = 0.1f; 
Glfloat farPlane = 1000.0f; 

glm::mat4 projectionMatrix = glm::perspective(FOV, width /height, nearPlane, farPlane); 
```

Note that `nearPlane` always needs to be greater than `0.0f` so that we can create the start of the frustum in front of the camera.

The `glm::perspective` function takes four parameters:

*   The `FOV`
*   The aspect ratio
*   The distance to the near plane
*   The distance to the far plane

So, after obtaining the projection matrix, we can finally perform a perspective projection transform on our view-transformed points to project the vertices onto the screen:

Position[final] = Projection[matrix] × View[matrix] × World[matrix] × Position[local]

Now that we understand this in theory, let's look at how we can actually implement this.

# Screen space

After multiplying the local position by the model, view, and projection matrix, OpenGL will transform the scene into screen space.

If the screen size of your application has a resolution of 1,280 x 720, then it will project the scene onto the screen like so; this is what can be seen by the camera in the view-space heading:

![](img/9c1e332d-5f27-4ab0-892f-474f4fdbfd81.png)

For this example, the width of the window will be 1,280 pixels and the height of the window will be 720 pixels.

# Render pipeline

As I mentioned earlier, we have to convert 3D objects that are made up of vertices and textures and represent them on a 2D screen as pixels on the screen. This is done with what is called a render pipeline. The following diagram explains the steps involved:

![](img/e540fba9-fd41-41eb-9526-6d433e8ff8a1.png)

A pipeline is simply a series of steps that are carried out one after the other to achieve a certain objective. The stages that are highlighted in the orange boxes (or lightly shaded boxes, if you're reading a black and white copy of this book) in the preceding diagram are fixed, meaning that you cannot modify how the data in the stages is processed. The stages in the blue or darker boxes are programmable stages, meaning that you can write special programs to modify the output. The preceding diagram shows a basic pipeline, which includes the minimum stages we need to complete to render objects. There are other stages, such as Geometry, Tessellation, and Compute, which are all optional stages. However, these will not be discussed in this book since we are only introducing graphics programming.

The graphics pipeline itself is common for both OpenGL and Vulkan. However, their implementation is different, but we will see this in the upcoming chapters.

The render pipeline is used to render 2D or 3D objects onto a TV or monitor. Let's look at each of the stages in the graphics pipeline in detail.

# Vertex specification

When we want to render an object to the screen, we set information regarding that object. The information that we will need to set is very basic, that is, the points or vertices that will make up the geometry. We will be creating the object by creating an array of vertices. This will be used to create a number of triangles that will make up the geometry we want to render. These vertices need to be sent to the graphics pipeline.

To send information to the pipeline in OpenGL, for example, we use **vertex array objects** (**VAO**) and **vertex buffer objects** (**VBO**).VAOs are used to define what data each vertex has; VBOs have the actual vertex data. 

Vertex data can have a series of attributes. A vertex could have property attributes, such as position, color, normal, and so on. Obviously, one of the main attributes that any vertex needs to have is the position information. Apart from the position, we will also look at other types of information that can be passed in, such as the color of each vertex. We will look at a few more attributes in future chapters in [Section 3](e10cd758-82e9-4314-96d2-f3b93f90aca4.xhtml), *Modern OpenGL 3D Game Development,* when we cover rendering primitives using OpenGL.

Let's say that we have an array of three points. Let's look at how we would create the `VAO` and the `VBO`:

```cpp
 float vertices[] = { 
    -0.5f, -0.5f, 0.0f, 
     0.5f, -0.5f, 0.0f, 
     0.0f,  0.5f, 0.0f 
};   
```

So, let's begin: 

1.  First, we generate a vertex array object of the Glint type. OpenGL returns a handle to the actual object for future reference, as follows:

```cpp
unsigned int VAO; 
glGenVertexArrays(1, &VAO);
```

2.  Then, we generate a vertex buffer object, as follows:

```cpp
unsigned int VBO; 
glGenBuffers(1, &VBO);   
```

3.  Next, we specify the type of buffer object. Here, it is of the `GL_ARRAY_BUFFER` type, meaning that this is an array buffer:

```cpp
glBindBuffer(GL_ARRAY_BUFFER, VBO);   
```

4.  Then, we bind the data to the buffer, as follows:

```cpp
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);  
```

The first parameter is a type of data buffer, which is of the `GL_ARRAY_BUFFER` type. The second parameter is the size of the data type that was passed in. `sizeof()` is a C++ keyword that gets the size of the data in bytes.

The next parameter is the data itself, while the last parameter is used to specify whether this data will change. `GL_STATIC_DRAW` means that the data will not be modified once the values have been stored.

5.  Then, we specify the vertex attributes, as follows:

*   The first parameter is the index of the attribute. In this case, we just have one attribute that has a position located at the 0th index.
*   The second is the size of the attribute. Each vertex is represented by three floats for *x*, *y*, and *z*, so here, the value that's being specified is `3`.
*   The third parameter is the type of data that is being passed in, which is of the `GL_FLOAT` type.
*   The fourth parameter is a Boolean asking whether the value should be normalized or used as is. We are specifying `GL_FALSE` since we don't want the data to be normalized.
*   The fifth parameter is called the stride; this specifies the offset between the attributes. Here, the value for the next position is the size of three floats, that is, *x*, *y*, and *z*.
*   The final parameter specifies the starting offset of the first component, which is 0 here. We are typecasting the data to a more generic data type (`void*`), called a void pointer:

```cpp
 glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 *  
 sizeof(float), (void*)0);  
```

6.  Finally, we enable the attribute at the 0th index, as follows:

```cpp
glEnableVertexAttribArray(0);
```

This is a basic example. This changes when we add additional attributes, such as color.

# Vertex shader

The vertex shader stage performs operations on a per-vertex basis. Depending on the number of vertices you have passed into the vertex shader, the vertex shader will be called that many times. If you are passing three vertices to form a triangle, then the vertex shader will be called three times.

Depending on the attributes that are passed into the shader, the vertex shader modifies the value that is passed in and outputs the final value of that attribute. For example, when you pass in a position attribute, you can manipulate its value and send out the final position of that vertex at the end of the vertex shader.

The following code is for a basic vertex shader for the single attribute that we passed in previously:

```cpp
#version 430 core 
layout (location = 0) in vec3 position; 

void main() 
{ 
    gl_Position = vec4(position.x, position.y, position.z, 1.0); 
} 
```

Shaders are programs written in a language similar to C. The shader will always begin with the version of the **GL Shader Language** (**GLSL**). There are other shader languages, including the **High-Level Shading Language** (**HLSL**), which is used by Direct 3D and CG. CG is also used in Unity.

Now, we will declare all the input attributes that we want to use. For this, we use the `layout` keyword and specify the location or the index of the attribute we want to use in brackets. Since we passed in one attribute for the vertex position and specified the index for it as 0 while specifying the attribute pointer, we set the location as 0\. Then, we use the `in` keyword to say that we are receiving the information. We store each position's value in a variable type called `vec3`, along with a name position.

`vec3`, which is a variable type, is used for vectors that are GLSL-intrinsic data types that can store data that's been passed into the shader. Here, since we are passing in position information that has an *x*, *y*, and *z* component, it's convenient to use a `vec3`. We also have a `vec4`, which has an additional `w` component that can be used to store color information, for example.

Each shader needs to have a main function in which the major function that's relevant to the shader will be performed. Here, we are not doing anything too complicated: we're just getting the `vec3`, converting it into a `vec4`, and then setting the value to `gl_Position`. We have to convert `vec3` into `vec4` since `gl_Position` is a `vec4`. It is also GLSL's intrinsic variable, which is used to store and output values from the vertex shader. 

Since this is a basic example of a vertex shader, we are not going to multiply each point with the `ModelViewProjection` matrix to transform the point onto the projection space. We will expand on this example later in this book.

# Vertex post-processing

At this stage, clipping occurs. Objects that are not in the visible cone of the camera are not rendered to the screen. These primitives, which are not displayed, are said to be clipped.

Let's say that only part of a sphere is visible. The primitive is broken into smaller primitives and only the primitives that are visible will be displayed. The vertex positions of the primitives are transformed from the clip space into the window space.

For example, in the following diagram, only parts of the sphere, triangle, and cuboid are visible. The remaining parts of the shapes are not visible to the camera and so they have been clipped:

![](img/23c03293-0fcc-4a7f-827d-f9abaf3df9c1.png)

# Primitive assembly

The primitive assembly stage gathers all the primitives that weren't clipped in the previous stage and creates a sequence of primitives.

Face culling is also performed at this stage. Face culling is the process in which primitives that are in front of the view but are facing backward will be culled since they won't be visible. For example, when you are looking at a cube, you only see the front face of the cube and not the back of the cube, so there is no point in rendering the back of the cube when it is not visible. This is called back-face culling.

# Rasterization

The GPU needs to convert the geometry that's been described in terms of vectors into pixels. We call this rasterization. The primitives that pass the previous stages of clipping and culling will be processed further so that they can be rasterized. The rasterization process creates a series of fragments from these primitives. The process of converting a primitive into a rasterized image is done by the GPU. In the process of rasterization (vector to pixel), we always lose information, hence the name f*ragments of primitives*. The fragment shader is used to calculate the final pixel's color value, which will be output to the screen.

# Fragment shader

Fragments from the rasterization stage are then processed using the fragment shader. Just like the vertex shader stage, the fragment shader is also a program that can be written so that we can perform modifications on each fragment.

The fragment shader will be called for each fragment from the previous stage.

Here is an example of a basic fragment shader:

```cpp
#version 430 core 
out vec4 Color; 

void main() 
{ 
    Color = vec4(0.0f, 0.0f, 1.0f, 1.0f); 
} 

```

Just like the vertex shader, you need to specify the GLSL version to use.

We use the `out` keyword to send the output value from the fragment shader. Here, we want to send out a variable of the `vec4` type called `Color`. The main function is where all the magic happens. For each fragment that gets processed, we set the value of `Color` to `blue`. So, when the primitive gets rendered to the viewport, it will be completely blue.

This is how the sphere becomes blue.

# Per-sample operation

In the same way that the vertex post-processing stage clipped a primitive, the per-sample operation also removes fragments that won't be shown. Whether a fragment needs to be displayed on the screen depends on certain tests that can be enabled by the user.

One of the more commonly used tests is the depth test. When enabled, this will check whether a fragment is behind another fragment. If this is the case, then the current fragment will be discarded. For example, here, only part of the cuboid is visible since it is behind the grey sphere:

![](img/590f5164-1edf-4d8c-864d-5eceddc171cf.png)

There are other tests we can perform as well, such as scissor and stencil tests, which will only show a portion of the screen or object based on certain conditions that we specify.

Color blending is also done at this stage. Here, based on certain blending equations, colors can be blended. For example, here, the sphere is transparent, so we can see the color of the cuboid blending into the color of the sphere:

![](img/98373bc0-12bf-46ac-b338-adebee132395.png)

# Framebuffer

Finally, when a per-sample operation is completed for all the fragments in a frame, the final image is rendered to the framebuffer, which is then presented on the screen.

A framebuffer is a collection of images that are drawn per frame. Each of these images is attached to the framebuffer. The framebuffer has attachments, such as the color image that's shown on the screen. There are also other attachments, such as the depth or the image/texture; this just stores the depth information of each pixel. The end user never sees this, but it is sometimes used for graphical purposes by games.

In OpenGL, the framebuffer is created automatically at the start. There are also user-created framebuffers that can be used to render the scene first, apply post-processing to it, and then hand it back to the system so that it can displayed on the screen.

# Summary

In this chapter, we covered some of the basics of mathematics that we will be using throughout this book. In particular, we learned about coordinate systems, points, vectors, and matrices. Then, we learned how to apply these concepts to Open GL mathematics and space transforms. Afterward, we looked at GLM, which is a math library that we will be using to make our mathematic calculations easier. Finally, we covered space transforms and understood the flow of the graphics pipeline. 

In the next chapter we will look at how to use a simple framework like SFML to make a 2D game