// File: sequential_shooter.c
// Author: James Miners-Webb
// Description: This program is the initial POC for my 3-Dimensional AimBot / Projectile interception
//               program. 
// Resources Used and Links:
// - Inspiriation for project: https://www.youtube.com/watch?v=aKd32I0uwAQ
// - Reference for math and inspiration: https://docs.google.com/document/d/1TKhiXzLMHVjDPX3a3U0uMvaiW1jWQWUmYpICjIDeMSA/edit
// - 3D vector research: https://www.superprof.co.uk/resources/academic/maths/analytical-geometry/vectors/3d-vectors.html
#include <stdio.h>

double calculateDisplacement();

int main(){
    printf("Hello World!\n");

    double px = calculateDisplacement();
    double py = calculateDisplacement();
    double pz = calculateDisplacement();
    return 0;
}


// The basis of this function is p = p + vt + (1/2)at^2
// "This expression describes displacement (p) over time given initial position (p), velocity (v) and acceleration (a)."
// This function will be called 3 times to get the p values of all 3 dimensions


double calculateDisplacement(){

    return 0.0;
}