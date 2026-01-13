

module stl() {
 //import("pawn.stl");
 //import("out.stl");
 //import("/home/alan/sync/m2/3dprint/pro-controller/pro-controller-decimated0.1.stl");
 import("/home/alan/sync/m2/3dprint/pro-controller/procon-shell0.1.stl");
}


difference() {
 stl();
 translate([0, -10, -10]) cube([20, 20, 20]);
}