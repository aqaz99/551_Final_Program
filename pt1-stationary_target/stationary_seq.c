// File: stationary_seq.c
// Author: James Miners-Webb
// Description: This program is a seqential version of calculating how to hit a stationary target 
//               given initial velocity and distance to target. 
//              The method is to 'brute force' all possible angles that the 'cannon' can take from 0-90 degrees.
//              We will then find the function given our knowns and current angle and integrate it using Left Riemann Sum
//               to find the total distance and see if that is equivalent to our target distance. 
// Resources Used and Links:

#include <stdio.h>
#include <math.h> // pow()
#include <assert.h>
#include <stdlib.h>

double f(double x, int initialHeight, double angleInDegrees, double projectileVelocity);
double projectileTravelTime(double distance, double angle, double projectileVelocity);

int main(int argc, char* argv[]){
    if(argc < 3){
        printf("Usage: ./stationary_seq target_distance initial_projectile_velocity\n");
        exit(1);
    }

    double targetDistance = atof(argv[1]);
    double projectileVelocity = atof(argv[2]);
    double initialProjectileHeight = 0;

    // Initial variables
    double stepSize = 100000;
    double deltax = targetDistance/stepSize;
    double errorToleranceToHit = 0.00001;
    // printf("Delta x is %f\n", deltax);

    for(int angle=0; angle<90; angle++){
        // Variables for storing Riemann Sum values
        double x = 0;
        double projectileDistanceTraveled = 0;
        double angleInDegrees = angle *  (M_PI / 180.0);


        // printf("tan of angle = %f\n", tan(angleInDegrees));
        while(1){
            double areaUnderSlice = f(x, initialProjectileHeight, angleInDegrees, projectileVelocity) * deltax;
            // If area under slice is negative, shot cannot reach
            if(areaUnderSlice < 0.0){
                // printf("slice: %f, Target cannot be reached with angle %d\n",areaUnderSlice, angle);
                break;
            }
            projectileDistanceTraveled += areaUnderSlice;

            if( (targetDistance - projectileDistanceTraveled) <= errorToleranceToHit){
                printf("Under slice: %f\n", areaUnderSlice);
                double travelTime = projectileTravelTime(targetDistance, angleInDegrees, projectileVelocity);
                printf("-- Hit Target! --\nProjectile traveled %f meters in %f seconds with angle %d degrees.\n", projectileDistanceTraveled, travelTime, angle);
                
                break;
            }

            x += deltax;
        }
    }

    return 0;
}


double f(double x, int initialHeight, double angleInDegrees, double projectileVelocity){
    double y;
    double underTheDivision = (2 * projectileVelocity * projectileVelocity * cos(angleInDegrees) * cos(angleInDegrees));

    // I should remove these calculations outside of the function to save cycles
    // y = h + x * tan(α) - g * x² / (2 * V₀² * cos²(α)) // 4.9 because gravity is divided by 2
    y = initialHeight + x * tan(angleInDegrees) - (9.8 * x * x / underTheDivision);

    // printf("x:%f y:%f\n",x, y);
    return y;
}

double projectileTravelTime(double distance, double angle, double projectileVelocity){
    // distance = rate * time
    // Total travel time -> Time = distance(x) / rate(cos(theta) * velocity)
    return (distance / (cos(angle) * projectileVelocity));
}