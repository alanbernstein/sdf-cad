package main

import (
	. "github.com/deadsy/sdfx/sdf"
)

func main() {
	RenderSTL(miniFanBracket(), 200, "fanBracket.stl")
}

func miniFanBracket() SDF3 {
	round := 1.5

	//body_3d := Cylinder3D(20, 26.3, 0)
	b1 := Box3D(V3{15, 45, 5}, round) // TODO set vs code's go.vetFlags to "-composites" to disable this check
	//b2 := Tz(Box3D(V3{15, 15, 10}, round), 2.5)
	b2 := Ry(Cylinder3D(15, 7.5, round), 90)
	b2 = Cut3D(b2, V3{0, 0, 0}, V3{0, 0, 1})

	hole1 := Ty(Cylinder3D(10, 2.5, 0), 15)  // screw hole 1
	hole2 := Ty(Cylinder3D(10, 2.5, 0), -15) // screw hole 2
	hole3 := Ry(Cylinder3D(20, 2.5, 0), 90)  // leg hole
	hole3b := Tz(Box3D(V3{15, 5, 5}, 0), -2.5)
	hole3 = U(hole3, hole3b)

	body := U(b1, b2)
	//body.(*UnionSDF3).SetMin(PolyMin(1.0))
	body = D(body, hole1)
	body = D(body, hole2)
	body = D(body, hole3)
	return body
}
