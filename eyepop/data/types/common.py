"""Common/shared types for the EyePop Data API."""

from typing import List

from pydantic import BaseModel


class Point2d(BaseModel):
    x: float
    y: float


class Point3d(Point2d):
    z: float | None = None


class Box(BaseModel):
    topLeft: Point2d
    bottomRight: Point2d


class Contour(BaseModel):
    points: List[Point2d]
    cutouts: List[List[Point2d]]


class Mask(BaseModel):
    bitmap: str
    width: int
    height: int
    stride: int
