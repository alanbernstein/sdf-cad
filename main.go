package main

import (
	. "github.com/deadsy/sdfx/sdf"
)

func main() {
	RenderSTL(miniFanBracket(), 200, "stl/fanBracket.stl")
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
