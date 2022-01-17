# Vectors
Vectors is a simple library toolkit dealing with common vector and point logic
in the 3-dimensional space.

Supports commonly used vector math functions including:
  * Vector magnitude
  * Addition with another vector or a real number.
  * Multiplication by another vector or a real number.
  * Dot product
  * Cross/scalar product
  * Create a unit vector of a vector
  * Angle between vectors
  * Check if two vectors are perpendicular, parallel or non-parallel

# Installation

```
pip install vectors
```

# Documentation

## Usage
There are multiple ways to create our vector instances using the vectors module.

We can first initialize some vectors and points calling their respective class
constructors as follows.

```Python
from vectors import Point, Vector

v1 = Vector(1, 2, 3) #=> Vector(1, 2, 3)
v2 = Vector(2, 4, 6) #=> Vector(2, 4, 6)

p1 = Point(1, 2, 6) #=> Point(1, 2, 6)
p2 = Point(2, 0, 2) #=> Point(2, 0, 2)
```

We can also create a Point instance or a Vector instance with a list
using the class method from_list().

```Python
from vectors import Vector
components = [1.2, 2.4, 3.8]

v = Vector.from_list(components) #=> Vector(1.2, 2.4, 3.8)
```

We can also create our Vectors from two Point instances using the classmethod
from_points().

```Python
from vectors import Point, Vector
p1 = Point(1, 2, 6) #=> Point(1, 2, 6)
p2 = Point(2, 0, 2) #=> Point(2, 0, 2)

v = Vector.from_points(p1, p2) #=> Vector(1, -2, -4)
```

We can also get access to the vector array to use it with other libraries.

```Python
from vectors import Vector
v = Vector(1,2,3)

v.to_points() #=> [1, 2, 3]
```

## Spherical

## Cylindrical


## Magnitude

We can get the magnitude of the vector easily.

```Python
from vectors import Vector
v = Vector(2,0,0)
v.magnitude() #==> 2.0

v1 = Vector(1,2,3)
v1.magnitude() #==> 3.7416573867739413

v2 = Vector(2,4,6)
v2.magnitude() #==> 7.483314773547883
```

## Addition

We can add a real number to a vector or compute the vector sum of two
vectors as follows.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

v1.add(2)  #=> Vector(3.0, 4.0, 5.0)
v1 + 2     #=> Vector(3.0, 4.0, 5.0)

v1.sum(v2) #=> Vector(3.0, 6.0, 9.0)
v1 + v2    #=> Vector(3.0, 6.0, 9.0)
```
Both methods return a Vector instance.


## Subtract


## Multiplication

We can multiply a vector by a real number.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

v1.multiply(4) #=> Vector(4.0, 8.0, 12.0)
# TODO: add * operator for real numbers
```
The above returns a Vector instance.

## Dot Product

We can find the dot product of two vectors.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

v1.dot(v2) #=> 28
```
We can also use angle theta on the dot function.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

# TODO: This does not work
v1.dot(v2, 180)
```
Dot product returns a real number.

## Cross/Scalar Product

We can find the cross product of two vectors.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

v1.cross(v2) #=> Vector(0, 0, 0)
```
Cross product returns a Vector instance, which is always perpendicular to the
other two vectors.

## Unit Vector

We can find the unit vector of a given vector.

```Python
from vectors import Vector
v1 = Vector(1,2,3)

v1.unit() #=> Vector(0.267261241912, 0.534522483825, 0.801783725737)
```
Unit vector function returns a Vector instance that has a magnitude of 1.

## Angle Theta

We can also find the angle theta between two vectors.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

v1.angle(v2) #=> 0.0
```
Angle is a measured in degrees.

## Rotate

## Parallel, Perpendicular, Non-Parallel

We can check if two vectors are parallel, perpendicular or non-parallel to each
other.

```Python
from vectors import Vector
v1 = Vector(1,2,3)
v2 = Vector(2,4,6)

v1.parallel(v2) #=> True
v1.perpendicular(v2) #=> False
v1.non_parallel(v2) #=> False
```
All of the above return either True or False.

#TODO
  * Create Analytic Geometry Toolkit based on the vectors toolkit.

----

**I'm looking for collaborators, so if you have something interesting, feel free
to collaborate.**






