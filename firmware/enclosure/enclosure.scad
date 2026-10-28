// Enclosure for Boolian's ESP32-C5 microphone and motion sensor.
//
// The board hangs by its ends (its header pins point down and the jumper wires need room below): a ledge under the
// USB end, with both USB-C ports lined up with a slot in the right wall, and a stepped post under the antenna end.
// The USB plug holds its end down; a peg on the lid holds the other.
// The INMP441 sits in a ring behind a small sound hole in the front wall, and the AM312's dome pokes through a hole
// above it. The lid is a friction fit.
//
// Measure your boards and fix the numbers marked MEASURE, then:
//   openscad -D part=\"box\" -o box.stl enclosure.scad
//   openscad -D part=\"lid\" -o lid.stl enclosure.scad
// Print the box open side up and the lid flat side down; no supports needed.

part = "both"; // box | lid | both

/* [Boards: MEASURE] */
board_l = 54;      // ESP32-C5 DevKitC-1, USB end to antenna end
board_w = 26;      // across the pin headers
board_t = 1.6;     // PCB thickness
usb_slot_w = 23;   // both USB-C ports, side by side on the short edge
usb_slot_h = 6.8;    // room for the plug's moulding, not just the socket
usb_above_pcb = 2.5; // height of the socket's centre above the PCB's top face
mic_pcb_d = 14.2;  // round INMP441 breakout
pir_dome_d = 12.2; // AM312 lens dome

/* [Box] */
wall = 2.2;
inner_w = 74;      // left-right (along the board)
inner_d = 60;      // front-back (sensors are in the front wall; their jumpers need ~25 mm behind it)
inner_h = 42;
board_z = 24;      // underside of the board above the floor: leaves room for the jumpers to bend
fit = 0.3;         // clearance for the lid lip and the board slots
corner_r = 3;
mount_tabs = true; // screw tabs on the back for fixing to a post or a wall

/* [Sensors] */
sensor_x = 20;     // centre of both sensors from the inside of the left wall
mic_z = 13;        // mic centre above the floor
pir_z = 31;        // PIR centre above the floor
sound_hole_d = 3;

$fn = 64;

module rounded_box(w, d, h, r) {
    hull() for (x = [r, w - r], y = [r, d - r]) translate([x, y, 0]) cylinder(r = r, h = h);
}

module box() {
    ow = inner_w + 2 * wall; od = inner_d + 2 * wall; oh = inner_h + wall;
    board_x0 = board_x0();  // the board's USB end touches the right wall
    board_cy = board_cy();  // towards the back, clear of the sensors' wiring
    difference() {
        union() {
            rounded_box(ow, od, oh, corner_r);
            if (mount_tabs) for (x = [8, ow - 8]) translate([x, od, 0]) difference() {
                hull() {
                    translate([-7, -1, 0]) cube([14, 1, 4]);
                    translate([0, 7, 0]) cylinder(r = 7, h = 4);
                }
                translate([0, 7, -1]) cylinder(d = 4.2, h = 6);
            }
        }
        translate([wall, wall, wall]) cube([inner_w, inner_d, inner_h + 1]);
        // USB slot, centred on the board's sockets
        translate([ow - wall - 1, board_cy - usb_slot_w / 2, wall + board_z + board_t + usb_above_pcb - usb_slot_h / 2])
            cube([wall + 2, usb_slot_w, usb_slot_h]);
        // sound hole, flared outside so it doesn't whistle
        translate([wall + sensor_x, -1, wall + mic_z]) rotate([-90, 0, 0]) {
            cylinder(d = sound_hole_d, h = wall + 2);
            cylinder(d1 = sound_hole_d + 3, d2 = sound_hole_d, h = 1 + wall / 2);
        }
        // PIR dome
        translate([wall + sensor_x, -1, wall + pir_z]) rotate([-90, 0, 0]) cylinder(d = pir_dome_d + fit, h = wall + 2);
    }
    // ring that centres the mic's PCB over the sound hole (glue it in)
    translate([wall + sensor_x, wall, wall + mic_z]) rotate([-90, 0, 0]) difference() {
        cylinder(d = mic_pcb_d + fit + 3, h = 3);
        translate([0, 0, -1]) cylinder(d = mic_pcb_d + fit, h = 5);
    }
    // ledge under the USB end: narrow enough to miss the header pins at the corners
    translate([wall + inner_w - 5, board_cy - 8, wall]) cube([5, 16, board_z]);
    // stepped post under the antenna end: the step stops the board sliding away from the USB slot
    translate([board_x0 - 4 - fit, board_cy - 6, wall]) difference() {
        cube([8, 12, board_z + board_t + 4]);
        translate([4, -1, board_z]) cube([5, 14, 10]);
    }
}

function board_x0() = wall + inner_w - board_l;
function board_cy() = wall + inner_d - board_w - 6 + board_w / 2;

module lid() {
    ow = inner_w + 2 * wall; od = inner_d + 2 * wall;
    rounded_box(ow, od, wall, corner_r);
    translate([wall + fit, wall + fit, 0]) difference() {
        cube([inner_w - 2 * fit, inner_d - 2 * fit, wall + 4]);
        translate([1.6, 1.6, 0]) cube([inner_w - 2 * fit - 3.2, inner_d - 2 * fit - 3.2, wall + 5]);
    }
    // peg that holds the board's antenna end down (the lid goes on upside down, so y is mirrored)
    translate([board_x0() + 3, od - board_cy(), 0]) cylinder(d = 5, h = wall + inner_h - board_z - board_t - fit);
}

if (part == "box" || part == "both") box();
if (part == "lid") lid();
if (part == "both") translate([0, -inner_d - 2 * wall - 25, 0]) lid();
