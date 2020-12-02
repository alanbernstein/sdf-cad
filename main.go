package main

import (
	"os"

	. "github.com/deadsy/sdfx/sdf"
)

func main() {
	defaultModel := "mackeyboard"
	var model string
	if len(os.Args) == 1 {
		model = defaultModel
	} else {
		model = os.Args[1]
	}

	switch model {
	case "minifan":
		RenderSTL(miniFanBracket(), 200, "stl/fanBracket.stl")
	case "audiojack":
		RenderSTL(audioJackBracket(), 200, "stl/audioJackBracket.stl")
	case "mackeyboard":
		RenderSTL(macKeyboardClamp2(), 200, "stl/macKeyboardClamp.stl")
	case "test":
		RenderSTL(test(), 200, "stl/test.stl")
		RenderSTL(box_test(), 200, "stl/box_test.stl")
	}
}

func miniFanBracket() SDF3 {
	// figured this out from sdfx example function cc18c

	round := 1.5
	d1 := 5.0 // diameter of fan leg

	//body_3d := Cylinder3D(20, 26.3, 0)
	b1 := Box3D(V3{15, 45, 5}, round)
	//b2 := Tz(Box3D(V3{15, 15, 10}, round), 2.5)
	b2 := Ry(Cylinder3D(15, 7.5, round), 90)
	b2 = Cut3D(b2, V3{0, 0, 0}, V3{0, 0, 1})

	hole1 := Ty(Cylinder3D(10, 1.5, 0), 15)  // screw hole 1
	hole2 := Ty(Cylinder3D(10, 1.5, 0), -15) // screw hole 2
	hole3 := Ry(Cylinder3D(20, d1/2, 0), 90) // leg hole
	hole3b := Tz(Box3D(V3{15, 5, 5}, 0), -2.5)
	hole3 = U(hole3, hole3b)

	body := U(b1, b2)
	//body.(*UnionSDF3).SetMin(PolyMin(1.0))
	body = D(body, hole1)
	body = D(body, hole2)
	body = D(body, hole3)
	return body
}

func audioJackBracket() SDF3 {
	d1 := 9.5  // jack outer diameter
	d2 := 6.0  // jack metal diameter
	l1 := 35.0 // full length of plastic jack
	//l2 := 20.0 // length of plastic jack before slant
	//w := 4.5   // width of thin part of plastic jack

	jack := Cylinder3D(l1, d1/2, 0)
	b1 := Tz(Cylinder3D(15, d2/2, 0), -1)
	jack = U(jack, b1)

	return jack
}

const keyHeight = 3.0 - 0.0 // reduce print time a bit
const keyWidth = 15.0 + 2.0
const keyLengthShift = 39.5 + 1.0
const keyLengthEnter = 30.0 + 1.0
const keySpacing = 3.0 - 1.5 // 0.5 fudge
const topThickness = 1.0

const twoD = true // use 2d rounded corners
const side = 1.0  // 1.0 for left, -1.0 for right

const keyLengthDeltaHalf = side * (keyLengthShift - keyLengthEnter) / 2

// Box3DR defines a 3D box with corners rounded in 2D (like a mac laptop keyboard key).
// Extrusion is in the Z direction, outer dimensions are the same as the equivalent Box3D call
func Box3DR(p V3, rad2 float64) SDF3 {
	// return Box3D(p, rad2)
	return Extrude3D(Box2D(V2{p.X, p.Y}, rad2), p.Z)
}

func macKeyboardClamp() SDF3 {
	// simple 90-degree corners
	boxFunc := Box3D
	rad := 0.0
	if twoD {
		// interior corners rounded in 2D
		boxFunc = Box3DR
		rad = 1.0
	}
	shiftKeyNeg := T(boxFunc(V3{keyLengthShift, keyWidth, keyHeight}, rad), 0, 0, topThickness)
	enterKeyNeg := T(boxFunc(V3{keyLengthEnter, keyWidth, keyHeight}, rad), keyLengthDeltaHalf, keyWidth+keySpacing, topThickness)
	shiftKeyPos := T(boxFunc(V3{keyLengthShift + 2*keySpacing, keyWidth + 2*keySpacing, keyHeight}, rad), 0, 0, 0)
	enterKeyPos := T(boxFunc(V3{keyLengthEnter + 2*keySpacing, keyWidth + 2*keySpacing, keyHeight}, rad), keyLengthDeltaHalf, keySpacing+keyWidth, 0)

	fullPos := U(enterKeyPos, shiftKeyPos)
	// fullPos.(*UnionSDF3).SetMin(PolyMin(1.0)) // blending to get rounded corner - might need to do this in 2D
	fullCap := D(fullPos, shiftKeyNeg, enterKeyNeg)

	/*
		// TODO: add hinge extension. but i would need to integrate the clamp into the design as well...
		cylDiam := keyHeight
		cylLen := keyWidth + 1.5*keySpacing
		cylOffset := (keyLengthShift + 2*keySpacing) / 2
		hingeCylinder := Ty(Tx(Rx(Cylinder3D(cylLen, cylDiam/2, 0), 90), cylOffset), -.5)

		withHinge := U(fullCap, hingeCylinder)
		return withHinge
	*/
	return Rz(fullCap, 90)
}

func macKeyboardClamp2() SDF3 {
	// defines the positive components in 2D, so the external, concave, rounded corner is defined cleanly in 2D, then extruded
	rad2 := 1.0
	shiftKeyNeg := T(Box3DR(V3{keyLengthShift, keyWidth, keyHeight}, rad2), 0, 0, topThickness)
	enterKeyNeg := T(Box3DR(V3{keyLengthEnter, keyWidth, keyHeight}, rad2), keyLengthDeltaHalf, keyWidth+keySpacing, topThickness)
	shiftKeyPos := T2(Box2D(V2{keyLengthShift + 2*keySpacing, keyWidth + 2*keySpacing}, rad2), 0, 0)
	enterKeyPos := T2(Box2D(V2{keyLengthEnter + 2*keySpacing, keyWidth + 2*keySpacing}, rad2), keyLengthDeltaHalf, keySpacing+keyWidth)

	fullPos2d := U2(enterKeyPos, shiftKeyPos)
	fullPos2d.(*UnionSDF2).SetMin(PolyMin(rad2)) // blending to get rounded corner
	fullPos3d := Extrude3D(fullPos2d, keyHeight)
	fullCap := D(fullPos3d, shiftKeyNeg, enterKeyNeg)
	fullCap = Cut3D(fullCap, V3{(keyLengthShift)/2 + keySpacing, 0, 0}, V3{-1, 0, 0}) // slice off the little blob that results from PolyMin being applied where we don't want it
	// Cut3D(obj, passThroughPoint, normalVector)

	return fullCap
}

func test() SDF3 {
	//	a := Box3D(V3{2, 2, 2}, .5)
	//	b := Box3D(V3{1, 1, 1}, 0)
	//	return Difference3D(b, a)

	baseBox := Box3D(V3{30, 40, 30}, 0)
	innerBox := Offset3D(baseBox, 3)
	outerBox := Offset3D(baseBox, 6)
	box := Difference3D(outerBox, innerBox)

	lidZ := (0.75 - 0.5) * 30
	base := Cut3D(box, V3{0, 0, lidZ}, V3{0, 0, -1})
	// top := Cut3D(box, V3{0, 0, lidZ}, V3{0, 0, 1})

	return base

}

func box_test() SDF3 {
	// error conditions:
	// if outerRadius < wallThickness {
	// if sizeX < outerOfs {
	// if sizeY < outerOfs {
	// if sizeZ < outerOfs {

	const sizeX = 30.0
	const sizeY = 40.0
	const sizeZ = 30.0

	const wallThickness = 3.0
	const outerRadius = 6.0
	const lidPosition = 0.75 // 0..1 position of lid on box

	innerOfs := outerRadius - wallThickness
	outerOfs := innerOfs + wallThickness

	baseBox := Box3D(V3{sizeX - outerOfs, sizeY - outerOfs, sizeZ - outerOfs}, 0)
	innerBox := Offset3D(baseBox, innerOfs)
	outerBox := Offset3D(baseBox, outerOfs)
	box := Difference3D(outerBox, innerBox)

	lidZ := (lidPosition - 0.5) * sizeZ
	base := Cut3D(box, V3{0, 0, lidZ}, V3{0, 0, -1})
	// top := Cut3D(box, V3{0, 0, lidZ}, V3{0, 0, 1})

	return base
}

/*
func sdfBox3d(p, s V3) float64 {
func sdfBox3d(p, s V3) float64 {
func RevolveTheta3D(sdf SDF2, theta float64) SDF3 {
func Revolve3D(sdf SDF2) SDF3 {
func (s *SorSDF3) Evaluate(p V3) float64 {
func (s *SorSDF3) BoundingBox() Box3 {
func Extrude3D(sdf SDF2, height float64) SDF3 {
func TwistExtrude3D(sdf SDF2, height, twist float64) SDF3 {
func ScaleExtrude3D(sdf SDF2, height float64, scale V2) SDF3 {
func ScaleTwistExtrude3D(sdf SDF2, height, twist float64, scale V2) SDF3 {
func (s *ExtrudeSDF3) Evaluate(p V3) float64 {
func (s *ExtrudeSDF3) SetExtrude(extrude ExtrudeFunc) {
func (s *ExtrudeSDF3) BoundingBox() Box3 {
func ExtrudeRounded3D(sdf SDF2, height, round float64) SDF3 {
func (s *ExtrudeRoundedSDF3) Evaluate(p V3) float64 {
func (s *ExtrudeRoundedSDF3) BoundingBox() Box3 {
func Loft3D(sdf0, sdf1 SDF2, height, round float64) SDF3 {
func (s *LoftSDF3) Evaluate(p V3) float64 {
func (s *LoftSDF3) BoundingBox() Box3 {
func Box3D(size V3, round float64) SDF3 {
func (s *BoxSDF3) Evaluate(p V3) float64 {
func (s *BoxSDF3) BoundingBox() Box3 {
func Sphere3D(radius float64) SDF3 {
func (s *SphereSDF3) Evaluate(p V3) float64 {
func (s *SphereSDF3) BoundingBox() Box3 {
func Cylinder3D(height, radius, round float64) SDF3 {
func Capsule3D(radius, height float64) SDF3 {
func (s *CylinderSDF3) Evaluate(p V3) float64 {
func (s *CylinderSDF3) BoundingBox() Box3 {
func MultiCylinder3D(height, radius float64, positions V2Set) SDF3 {
func (s *MultiCylinderSDF3) Evaluate(p V3) float64 {
func (s *MultiCylinderSDF3) BoundingBox() Box3 {
func Cone3D(height, r0, r1, round float64) SDF3 {
func (s *ConeSDF3) Evaluate(p V3) float64 {
func (s *ConeSDF3) BoundingBox() Box3 {
func Transform3D(sdf SDF3, matrix M44) SDF3 {
func (s *TransformSDF3) Evaluate(p V3) float64 {
func (s *TransformSDF3) BoundingBox() Box3 {
func ScaleUniform3D(sdf SDF3, k float64) SDF3 {
func (s *ScaleUniformSDF3) Evaluate(p V3) float64 {
func (s *ScaleUniformSDF3) BoundingBox() Box3 {
func Union3D(sdf ...SDF3) SDF3 {
func (s *UnionSDF3) Evaluate(p V3) float64 {
func (s *UnionSDF3) SetMin(min MinFunc) {
func (s *UnionSDF3) BoundingBox() Box3 {
func Difference3D(s0, s1 SDF3) SDF3 {
func (s *DifferenceSDF3) Evaluate(p V3) float64 {
func (s *DifferenceSDF3) SetMax(max MaxFunc) {
func (s *DifferenceSDF3) BoundingBox() Box3 {
func Elongate3D(sdf SDF3, h V3) SDF3 {
func (s *ElongateSDF3) Evaluate(p V3) float64 {
func (s *ElongateSDF3) BoundingBox() Box3 {
func Intersect3D(s0, s1 SDF3) SDF3 {
func (s *IntersectionSDF3) Evaluate(p V3) float64 {
func (s *IntersectionSDF3) SetMax(max MaxFunc) {
func (s *IntersectionSDF3) BoundingBox() Box3 {
func Cut3D(sdf SDF3, a, n V3) SDF3 {
func (s *CutSDF3) Evaluate(p V3) float64 {
func (s *CutSDF3) BoundingBox() Box3 {
func Array3D(sdf SDF3, num V3i, step V3) SDF3 {
func (s *ArraySDF3) SetMin(min MinFunc) {
func (s *ArraySDF3) Evaluate(p V3) float64 {
func (s *ArraySDF3) BoundingBox() Box3 {
func RotateUnion3D(sdf SDF3, num int, step M44) SDF3 {
func (s *RotateUnionSDF3) Evaluate(p V3) float64 {
func (s *RotateUnionSDF3) SetMin(min MinFunc) {
func (s *RotateUnionSDF3) BoundingBox() Box3 {
func RotateCopy3D(
func (s *RotateCopySDF3) Evaluate(p V3) float64 {
func (s *RotateCopySDF3) BoundingBox() Box3 {
func AddConnector(sdf SDF3, connectors ...Connector3) SDF3 {
func (s *ConnectedSDF3) Evaluate(p V3) float64 {
func (s *ConnectedSDF3) BoundingBox() Box3 {
func Offset3D(sdf SDF3, offset float64) SDF3 {
func (s *OffsetSDF3) Evaluate(p V3) float64 {
func (s *OffsetSDF3) BoundingBox() Box3 {
*/
