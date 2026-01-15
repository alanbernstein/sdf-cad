include </home/alan/Dropbox/src/scad/libs/BOSL2/std.scad>;
include </home/alan/Dropbox/src/scad/libs/BOSL2/rounding.scad>;
$fn = 128;
in2mm = 25.4;

module procon_stl(S=1) {
 scale(S) 
//  import("/home/alan/sync/m2/3dprint/pro-controller/pro-controller-decimated0.05.stl");
//  import("/home/alan/sync/m2/3dprint/pro-controller/pro-controller-joycons-hulled-decimated0.1.stl");
 import("/home/alan/sync/m2/3dprint/pro-controller/procon-dec0.1-shell0.1-vox.05.stl");
}
// left_half(300) stl(in2mm);

module convexified_stl() {
    // used to identify parameters of cylinders to use for filtering out the vertices
    // in the joysticks that are "non-convex" (i.e. when considered as a unit separate from
    // the main controller body). Ultimately I just edited those vertices out manually in
    // Blender.
    d1 = .65;
    d2 = 1.03;
    union() {
     procon_stl();
     // #up(.833) left(1.641) ycyl(d=.37, h=3);
     up(.833) left(1.641) fwd(1.092) ycyl(d1=d1, d2=d2, h=.4, anchor=FRONT);
     // #up(.01) right(.798) ycyl(d=.44, h=3);
     up(.01) right(.798) fwd(1.092) ycyl(d1=d1, d2=d2, h=.4, anchor=FRONT);
    }
}
// convexified_stl();

module shell_cover() {
    difference() {
        down(12) xrot(-90) procon_stl(in2mm);
        #back(42) cuboid([200, 100, 50], anchor=BACK+TOP, rounding=12);
        #down(10.76) cuboid([12, 100, 8], rounding=4, edges="Y", anchor=FRONT);
    }
}
shell_cover();