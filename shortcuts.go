package main

import (
	. "github.com/deadsy/sdfx/sdf"
)

// U shorthand for Union3D
func U(sdf ...SDF3) SDF3 {
	return Union3D(sdf...)
}

// U2 shorthand for Union2D
func U2(sdf ...SDF2) SDF2 {
	return Union2D(sdf...)
}

// I shorthand for Intersect3D
func I(s0, s1 SDF3) SDF3 {
	return Intersect3D(s0, s1)
}

// D shorthand for Difference3D with any number of negative inputs
func D(s0 SDF3, ss ...SDF3) SDF3 {
	res := s0
	for _, s := range ss {
		res = Difference3D(res, s)
	}
	return res
}

// S shorthand for Scale3d
func S(obj SDF3, x, y, z float64) SDF3 {
	m := Scale3d(V3{X: x, Y: y, Z: z})
	return Transform3D(obj, m)
}

// T shorthand for Translate3d
func T(obj SDF3, x, y, z float64) SDF3 {
	m := Translate3d(V3{X: x, Y: y, Z: z})
	return Transform3D(obj, m)
}

// T2 shorthand for Translate2d
// TODO: check type of obj to consolidate T and T2
func T2(obj SDF2, x, y float64) SDF2 {
	m := Translate2d(V2{X: x, Y: y})
	return Transform2D(obj, m)
}

// Tx shorthand for applying Translate3d in the x direction.
func Tx(obj SDF3, x float64) SDF3 {
	m := Translate3d(V3{X: x, Y: 0, Z: 0})
	return Transform3D(obj, m)
}

// Ty shorthand for applying Translate3d in the y direction.
func Ty(obj SDF3, y float64) SDF3 {
	m := Translate3d(V3{X: 0, Y: y, Z: 0})
	return Transform3D(obj, m)
}

// Tz shorthand for applying Translate3d in the z direction.
func Tz(obj SDF3, z float64) SDF3 {
	m := Translate3d(V3{X: 0, Y: 0, Z: z})
	return Transform3D(obj, m)
}

// Rx shorthand for applying RotateX
func Rx(obj SDF3, deg float64) SDF3 {
	m := RotateX(DtoR(deg))
	return Transform3D(obj, m)
}

// Ry shorthand for applying RotateX
func Ry(obj SDF3, deg float64) SDF3 {
	m := RotateY(DtoR(deg))
	return Transform3D(obj, m)
}

// Rz shorthand for applying RotateX
func Rz(obj SDF3, deg float64) SDF3 {
	m := RotateZ(DtoR(deg))
	return Transform3D(obj, m)
}
