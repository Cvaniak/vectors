from __future__ import division

from functools import reduce
import math
from numbers import Real

def _set_in_list(values, index, new_value):
    values[index] = new_value
    
def _link_with_list(index):
    return property(lambda self: self._values[index],
                    lambda self, val: _set_in_list(self._values, index, val))

class Point:
    x = _link_with_list(0) 
    y = _link_with_list(1) 
    z = _link_with_list(2) 

    def __init__(self, x, y, z=0):
        self._values = [x, y, z]

    def __eq__(self, pt):
        return all([a == b for a, b in zip(self, pt)])

    def __iter__(self):
        for xyz in self._values:
            yield xyz

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def __contains__(self, item):
        return item in self._values

    def __repr__(self):
        return f"{self.__class__.__name__}{tuple(self._values)}"

    def __sub__(self, pt):
        """Return a Point instance as the displacement of two points."""
        if not isinstance(pt, type(self)):
            raise TypeError
        return self.__class__(*[a-b for a, b in zip(self, pt)])

    def __add__(self, pt):
        if not isinstance(pt, Point):
            raise TypeError
        return self.__class__(*[a+b for a, b in zip(self, pt)])

    def to_dict(self):
        return {"x":self.x, "y":self.y, "z":self.z}
    
    def to_list(self):
        '''Returns an array of [x,y,z] of the end points'''
        return self._values

    def xy(self):
        return (self.x, self.y)

    def xz(self):
        return (self.x, self.z)

    def yz(self):
        return (self.y, self.z)

    @classmethod
    def from_list(cls, l):
        """Return a Point instance from a given list"""
        if 2 <= len(l) <= 3:
            return cls(*l)
        raise AttributeError


class Vector(Point):
    """Vector class: Representing a vector in 3D space.

    Can accept formats of:
    Cartesian coordinates in the x, y, z space.(Regular initialization)
    Spherical coordinates in the r, theta, phi space.(Spherical class method)
    Cylindrical coordinates in the r, theta, z space.(Cylindrical class method)
    """

    def __init__(self, *args, **kwargs):
        return super(Vector, self).__init__(*args, **kwargs)

    def __add__(self, obj):
        """Add two vectors together"""
        if type(obj) is type(self):
            return self.__class__(*[a+b for a, b in zip(self, obj)])
        elif isinstance(obj, Real):
            return self.add(obj)
        raise TypeError

    def __sub__(self, obj):
        """Subtract two vectors"""
        if type(obj) is type(self):
            return self.__class__(*[a-b for a, b in zip(self, obj)])
        elif isinstance(obj, Real):
            return self.subtract(obj)
        raise TypeError

    def __mul__(self, obj):
        """Return a Vector instance as the cross product of two vectors"""
        if type(obj) is type(self):
            return self.cross(obj)
        elif isinstance(obj, Real):
            return self.__class__(*[value*obj for value in self])
        raise TypeError

    def __round__(self, n=None):
        if n is not None:
            return self.__class__(round(self.x, n), round(self.y, n), round(self.z, n))
        return self.__class__(round(self.x), round(self.y), round(self.z))

    def add(self, obj):
        return self + obj

    def subtract(self, obj):
        return self - obj

    def multiply(self, number):
        return self * number

    def magnitude(self):
        """Return magnitude of the vector."""
        return math.sqrt(sum([x ** 2 for x in self]))

    def sum(self, vector):
        return self + vector

    def dot(self, vector, theta=None):
        """Return the dot product of two vectors.

        If theta is given then the dot product is computed as
        v1*v1 = |v1||v2|cos(theta). Argument theta
        is measured in degrees.
        """
        if theta is not None:
            return (self.magnitude() * vector.magnitude() *
                    math.degrees(math.cos(theta)))
        return sum([a * b for a, b in zip(self, vector)])

    def cross(self, vector):
        """Return a Vector instance as the cross product of two vectors"""
        return self.__class__((self.y * vector.z - self.z * vector.y),
                      (self.z * vector.x - self.x * vector.z),
                      (self.x * vector.y - self.y * vector.x))

    def unit(self):
        """Return a Vector instance of the unit vector"""
        magnitude = self.magnitude()
        return self.__class__(*[value/magnitude for value in self])

    def angle(self, vector):
        """Return the angle between two vectors in degrees."""
        return math.degrees(
            math.acos(
                self.dot(vector) /
                (self.magnitude() * vector.magnitude())
            )
        )

    def parallel(self, vector):
        """Return True if vectors are parallel to each other."""
        return not any(self.cross(vector))

    def perpendicular(self, vector):
        """Return True if vectors are perpendicular to each other."""
        return not self.dot(vector)

    def non_parallel(self, vector):
        """Return True if vectors are non-parallel.

        Non-parallel vectors are vectors which are neither parallel
        nor perpendicular to each other.
        """
        if self.parallel(vector) or self.perpendicular(vector):
            return True
        return False

    def rotate(self, angle, axis=(0, 0, 1)):
        """Returns the rotated vector. Assumes angle is in radians"""
        if not all(isinstance(a, int) for a in axis):
            raise ValueError
        x, y, z = self.x, self.y, self.z

        # Z axis rotation
        if(axis[2]):
            x = (self.x * math.cos(angle) - self.y * math.sin(angle))
            y = (self.x * math.sin(angle) + self.y * math.cos(angle))

        # Y axis rotation
        if(axis[1]):
            x = self.x * math.cos(angle) + self.z * math.sin(angle)
            z = -self.x * math.sin(angle) + self.z * math.cos(angle)

        # X axis rotation
        if(axis[0]):
            y = self.y * math.cos(angle) - self.z * math.sin(angle)
            z = self.y * math.sin(angle) + self.z * math.cos(angle)

        return self.__class__(x, y, z)

    @classmethod
    def from_points(cls, point1, point2):
        """Return a Vector instance from two given points."""
        if isinstance(point1, Point) and isinstance(point2, Point):
            displacement = point1 - point2
            return cls(displacement.x, displacement.y, displacement.z)
        raise TypeError

    @classmethod
    def spherical(cls, mag, theta, phi=0):
        '''Returns a Vector instance from spherical coordinates'''
        return cls(
            mag * math.sin(phi) * math.cos(theta),  # X
            mag * math.sin(phi) * math.sin(theta),  # Y
            mag * math.cos(phi)  # Z
        )

    @classmethod
    def cylindrical(cls, mag, theta, z=0):
        '''Returns a Vector instance from cylindrical coordinates'''
        return cls(
            mag * math.cos(theta),  # X
            mag * math.sin(theta),  # Y
            z  # Z
        )
