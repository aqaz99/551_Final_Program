// File: stationary_seq.c
// Author: James Miners-Webb
// Description: This program is a seqential version of calculating how to hit a stationary target 
//               given initial velocity and distance to target. 
//              The method is to 'brute force' all possible angles that the 'cannon' can take from 0-90 degrees.
//              We will then find the function given our knowns and current angle and integrate it using Left Riemann Sum
//               to find the total distance and see if that is equivalent to our target distance. 
// Resources Used and Links:
//              - Results of output can be checked here for validity: https://amesweb.info/Physics/Projectile-Motion-Calculator.aspx

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
    
    // The dimensions below create an acceptable hit tolerance of a 4 inch x 4 inch square
    // Within 10 micrometers on the x axis
    double xToleranceToHit = 0.005;

    // Within 10 centimeters on the y axis
    double yToleranceToHit = 0.005;

    double maxProjectileDistance = 0.0;


    // Increase launch angle by a very small number as to cover many possible trajectories
    for(double angle=0; angle<90; angle += .005){
        // Variables for storing Riemann Sum values
        double x = 0;
        double projectileDistanceTraveled = 0;
        double angleInRadians = angle *  (M_PI / 180.0);


        while(1){
            double areaUnderSlice = f(x, initialProjectileHeight, angleInRadians, projectileVelocity) * deltax;
            
            // If area under slice is negative, shot cannot reach or if we haven't found solution by the distance of the target
            if(areaUnderSlice < 0.0 || x > targetDistance){
                break;
            }

            // Total distance in x direction
            projectileDistanceTraveled += areaUnderSlice;
            double projectileElevation = f(projectileDistanceTraveled, initialProjectileHeight, angleInRadians, projectileVelocity);

            // Y value is negative, can't hit target
            if(projectileElevation < 0.0){
                // printf("crossed x axis at %f\n", projectileDistanceTraveled);
                break;
            }

            // Candidate for hit, distance travelled by shot is almost equal to distance to target, now we need to check elevation
            if( fabs(targetDistance - projectileDistanceTraveled) <= xToleranceToHit){
                // printf("Difference between %f and %f: %f\n",targetDistance, projectileDistanceTraveled, fabs(targetDistance - projectileDistanceTraveled));

                // We can check projectile elevation by plugging into f() function with our distance as the x value now
                if( projectileElevation <= yToleranceToHit){
                    double travelTime = projectileTravelTime(targetDistance, angleInRadians, projectileVelocity);
                    printf("-- Hit Target! --\nProjectile traveled %f meters in %f seconds with angle %f degrees.\n", projectileDistanceTraveled, travelTime, angle);
                    printf("Projectile elevation: %f\n",projectileElevation);
                    break;
                }
            }

            x += deltax;
        }
        if(projectileDistanceTraveled > maxProjectileDistance){
            maxProjectileDistance = projectileDistanceTraveled;
        }
        // printf("Final distance for %f = %f\n", angle, projectileDistanceTraveled);
    }
    printf("Max projectile distance with initial velocity of %f is %f\n", projectileVelocity, maxProjectileDistance);
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