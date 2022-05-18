
// https://www.omnicalculator.com/physics/range-projectile-motion

#include <stdio.h>
#include <math.h> // pow()
#include <assert.h>
#include <stdlib.h>
#include "non_cuda_headers.h"

int main(int argc, char* argv[]){
    // CL Arguments
    if(argc !=  5){
        printf("Usage: ./moving_seq TARGET_velocity TARGET_initial_height TARGET_firing_angle INTERCEPTOR_velocity\n");
        exit(1);
    }

    double target_velocity = atof(argv[1]);
    double target_initial_height = atof(argv[2]);
    double target_firing_angle = atof(argv[3]);
    double interceptor_velocity = atof(argv[4]);

    // Less accurate than scientific and CUDA because it takes so long
    // Within .05 meters or ~2 inches
    double xToleranceToHit = 0.05;

    // Within .05 meters or ~2 inches
    double yToleranceToHit = 0.05;

    // Init moving target with initial velocity of 35 and firing angle of 45
    ProjectileClass* target = malloc(sizeof(ProjectileClass)); // Object I want to hit
    initProjectile(target, target_velocity, target_initial_height, target_firing_angle);
    printEquation(target);

    // Init interceptor with initial velocity of user input and both initial height and angle to zero
    ProjectileClass* interceptor = malloc(sizeof(ProjectileClass)); // Object I want to hit
    initProjectile(interceptor, interceptor_velocity, 0, 0);

    //print info about target projectile
    double total_distance_traveled = calculateTotalDistance(target);
    printf("Target final x position: %f\n", total_distance_traveled);
    double total_travel_time = timeGivenDistance(total_distance_traveled, target->firingAngle, target->initialVelocity);
    printf("Total travel time: %f\n", total_travel_time);


    // Lets say we want to hit our target at (travel time)/2 
    // 1. Find the x and y location of our target at that time
    // 2. Find an angle that an interceptor could be fired at that passes through that (x, y)
    // 3. Determine the time it would take for the interceptor to get to that location
    // 4. If the interceptor can reach that location in time, display when the interceptor needs to fire to hit the target

    // 1. Get (x, y) at total time / 2 
    printf("Attempting to hit target at %f\n", total_travel_time/2);
    double targetX = distanceGivenTime(total_travel_time/2, target->firingAngle, target->initialVelocity);
    double targetY = predictedYValue(target, targetX);
    printf("Targets position at %f seconds is (%f, %f)\n",total_travel_time/2, targetX, targetY);

    // Initial Riemann sum variables
    double stepSize = 15000;
    double deltax = targetX/stepSize;

    // Timing
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);


    // Stopper for once we find a solution, don't need to find more
    int foundSolution = 0;
    for(double angle=0.0; angle<90; angle+=.001){ // For all firing angles
        // Set new interceptor angle
        interceptor->firingAngle = angle;
        // Variables for storing Riemann Sum values
        double x = 0;
        double projectileDistanceTraveled = 0;
        while(foundSolution != 1){
            // Should this x be dist traveled?
            double areaUnderSlice = predictedYValue(interceptor, x) * deltax;
            
            // If area under slice is negative, shot cannot reach or if we haven't found solution by the distance of the target
            if(areaUnderSlice < 0.0){
                break;
            }

            // Total distance in x direction
            projectileDistanceTraveled += areaUnderSlice;
            double projectileElevation = predictedYValue(interceptor, projectileDistanceTraveled);

            // Get time it will take our projectile to hit the target
            double intercTimeToTarget = timeGivenDistance(targetX, interceptor->firingAngle, interceptor->initialVelocity);
            if(intercTimeToTarget > total_travel_time){ // Skip angles where projectile would travel too long to get there
                break;
            }

            // Y value is negative, can't hit target
            if(projectileElevation < 0.0){
                break;
            }
            // Candidate for hit, distance travelled by shot is almost equal to distance to target, now we need to check elevation
            if( fabs(targetX - projectileDistanceTraveled) <= xToleranceToHit){


                // Check to see if both y's are the same and if the time to target is positive, meaning the shot is possible. neg values are solutions but not in time
                if( fabs(projectileElevation - targetY) <= yToleranceToHit && ((total_travel_time/2) - intercTimeToTarget) > 0.0){
                    foundSolution = 1;
                    printf("-------Can hit target!------\n");
                    printf("- Angle: %f\n- Time to Target: %f seconds\n- Launch after: %f seconds\n", angle, intercTimeToTarget, (total_travel_time/2) - intercTimeToTarget);
                    printf("- Target(x, y): (%f, %f)\n- Interceptor(x, y): (%f, %f)\n- ",targetX, targetY, projectileDistanceTraveled, projectileElevation);
                    printEquation(interceptor);
                    printf("----------------------------\n");
                    angle+=.1; // Increment angle by a good chunk as to not get a giant list of firing solutions
                    break;
                }
            }
            x += deltax;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsedTime = (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("-- Total run time:  %f seconds --\n", elapsedTime);
    return 0;
}
