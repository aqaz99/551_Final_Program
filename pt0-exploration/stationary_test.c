// File: sequential_shooter.c
// Author: James Miners-Webb
// Description: This program is the initial POC for my 3-Dimensional AimBot / Projectile interception
//               program. 
// Resources Used and Links:
// - Inspiriation for project: https://www.youtube.com/watch?v=aKd32I0uwAQ
// - Reference for math and inspiration: https://docs.google.com/document/d/1TKhiXzLMHVjDPX3a3U0uMvaiW1jWQWUmYpICjIDeMSA/edit
// - 3D vector research: https://www.superprof.co.uk/resources/academic/maths/analytical-geometry/vectors/3d-vectors.html
// - 3D Distance formula example: https://www.engineeringtoolbox.com/distance-relationship-between-two-points-d_1854.html
// - Projectile Motion Calculator: https://www.omnicalculator.com/physics/projectile-motion
// - Trajectory Calculator: https://www.omnicalculator.com/physics/trajectory-projectile-motion
// - SSS Triangle solving: https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html

// - Pretty good: https://stackoverflow.com/questions/2248876/2d-game-fire-at-a-moving-target-by-predicting-intersection-of-projectile-and-u

// MOST HELPFUL LINK: https://www.forrestthewoods.com/blog/solving_ballistic_trajectories/
// SECOND MOST HELPFUL: https://casualhacks.net/blog/2019-09-17/projectile-solver/

#include <stdio.h>
#include <math.h> // pow()
#include <assert.h>
#include <stdlib.h>
#include <time.h> // Clocking speeds

typedef struct {
  double x, y, z;
} PointClass;


double getDistanceBetweenPoints(const PointClass*, const PointClass*);
double getFiringAngle(double, double, PointClass*);
double projectileTravelTime(double, double, double);
double getRotation(const PointClass*, const PointClass*, double);
void printEquation(double x, double initialHeight, double angleInDegrees, double projectileVelocity);

void init_point(PointClass*, double, double, double);

// In order to hit a stationary target, we need to know:
// initial projectile speed
// distance to target
// Then we can determine firing angle(s), there should be two on short arc and one long rainbow arc
int main(){
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    PointClass* cannon = malloc(sizeof(PointClass));
    PointClass* target = malloc(sizeof(PointClass));

    init_point(cannon, 0, 0, 0);
    init_point(target, 0, 0, 125);

    double distance = getDistanceBetweenPoints(cannon, target);
    printf("The distance between these points is %f\n", distance);

    double projectileVelocity = 45; // Meters per second

    double firingAngle = getFiringAngle(projectileVelocity, distance, target);

    printf("Firing angle is: %f degrees\n", (firingAngle * 180) / M_PI);
    double projectileTime = projectileTravelTime(distance, firingAngle, projectileVelocity);

    double cannonRotation = getRotation(target, cannon, distance);

    if(cannonRotation == cannonRotation){ //Check for nan
        printf("Theta/azimuth: %f degrees\n", cannonRotation);
    }else{
        printf("No rotation required\n");
    }

    printf("Shortest projectile travel time: %f\n", projectileTime);
    printEquation(distance, cannon->y, firingAngle, projectileVelocity);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsedTime = (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("-- Total run time:  %f seconds --\n", elapsedTime);
    return 0;
}


// Using the 3d distance formula d = ( (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 )^(1/2) 
// Verified with online too https://www.engineeringtoolbox.com/distance-relationship-between-two-points-d_1854.html
double getDistanceBetweenPoints(const PointClass* a,const PointClass* b){

    // distance kept on seperate lines for legibility
    double dist_x = pow((b->x - a->x), 2);
    double dist_y = pow((b->y - a->y), 2);
    double dist_z = pow((b->z - a->z), 2);

    // printf("x: %f\ny: %f\nz: %f\n",dist_x, dist_y, dist_z);

    // Sum distances
    double dist_sum = dist_x + dist_y + dist_z;

    // Return the distance sum to 1/2 power
    return pow(dist_sum, .5);
}

// Based on equation for calculating theta
// https://www.forrestthewoods.com/blog/solving_ballistic_trajectories/
double getFiringAngle(double projectileVelocity, double targetDistance, PointClass* target){
    // Gravity is not represented as negative here
    double gravity = 9.8;
    double x = targetDistance; // X or Z distance, doesn't matter. Only need 2d so we just take target distance
    double y = target->y;

    // Step one, handle the set up parenthesis under sqrt
    double theta = (gravity * x * x + (2 * projectileVelocity * projectileVelocity * y));
    
    // If the value under the square root is negative then there are no solutions as the target is too far away
    // Step two handle rest under sqrt, subtraction, possible source of error
    double negative_check = (projectileVelocity * projectileVelocity * projectileVelocity * projectileVelocity) - (gravity * theta);

    // printf("Value under the square root: %f\n", negative_check);
    if(negative_check <= 0.0){
        printf("Target is too far away, can't hit it\n");
        exit(0);
    }

    theta = sqrt(negative_check);

    double quadratic_plus = (projectileVelocity * projectileVelocity + theta)/(gravity * x);
    double quadratic_minus = (projectileVelocity * projectileVelocity - theta)/(gravity * x);

    // The smaller the value after arctan, the lower the angle, prevents bullet lobbing
    quadratic_plus = atan(quadratic_plus);
    quadratic_minus = atan(quadratic_minus);

    printf("The values of the quadratic formula thetas are (+)%f degrees and (-)%f degrees\n", (quadratic_plus * 180) / M_PI, (quadratic_minus * 180) / M_PI);

    // Check to make sure neither are Nan
    assert(quadratic_plus == quadratic_plus);
    assert(quadratic_minus == quadratic_minus);

    // Here we want to return the smaller angle value, it will be less of a lobbed path and have a quicker travel time
    if( ((quadratic_plus * 180) / M_PI ) < (quadratic_minus * 180) / M_PI){
        return quadratic_plus;
    }

    printf("--------------------\n");
    printf("Non-optimal Equation:\n");
    printEquation(targetDistance, 0, quadratic_plus, projectileVelocity);
    printf("--------------------\n");
    return quadratic_minus;
}

// Use the law of cosines, or SSS to get necessary rotation or azimuth for cannon
double getRotation(const PointClass* target, const PointClass* cannon, double distance){
    // Execute Law of Cosines to solve a SSS triangle:
    // https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
    // Formula is of form: Cos(A) = (b^2 + c^2 - a^2)/2bc
    double sideC = target->x - cannon->x;
    double sideB = distance;
    double sideA = target->z - cannon->z;

    double cosA = ((sideB * sideB) + (sideC * sideC) - (sideA * sideA))/(2 * sideB * sideC);
    cosA = acos(cosA);
    return (cosA * 180) / M_PI;
}

double projectileTravelTime(double distance, double angle, double projectileVelocity){
    // distance = rate * time
    // Total travel time -> Time = distance(x) / rate(cos(theta) * velocity)
    return (distance / (cos(angle) * projectileVelocity));
}

void init_point(PointClass* point, double x, double y, double z){
    point->x = x;
    point->y = y;
    point->z = z;
}

void printEquation(double x, double initialHeight, double angleInRadians, double projectileVelocity){
    double underTheDivision = (2 * projectileVelocity * projectileVelocity * cos(angleInRadians) * cos(angleInRadians));
    printf("Function equation F(x) = %f + %fx - (9.8x^2) / %f\n",initialHeight, tan(angleInRadians), underTheDivision);
}