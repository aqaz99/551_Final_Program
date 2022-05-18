
// https://www.omnicalculator.com/physics/range-projectile-motion

#include <stdio.h>
#include <math.h> // pow()
#include <assert.h>
#include <stdlib.h>
#include "non_cuda_headers.h"

int main(int argc, char* argv[]){
    // CL Arguments
    if(argc !=  5){
        printf("Usage: ./moving_scientific TARGET_velocity TARGET_initial_height TARGET_firing_angle INTERCEPTOR_velocity\n");
        exit(1);
    }

    double target_velocity = atof(argv[1]);
    double target_initial_height = atof(argv[2]);
    double target_firing_angle = atof(argv[3]);
    double interceptor_velocity = atof(argv[4]);

    // Much smaller y tolerance to hit because we use math to get the exact angle
    // Within .05 centimeters on the y axis
    double yToleranceToHit = 0.0005;

    // Init moving target with initial velocity of 35 and firing angle of 45
    ProjectileClass* target = malloc(sizeof(ProjectileClass)); // Object I want to hit
    initProjectile(target, target_velocity, target_initial_height, target_firing_angle);
    printEquation(target);

    // Init interceptor with initial velocity of user input and both initial height and angle to zero
    ProjectileClass* interceptor = malloc(sizeof(ProjectileClass)); // Object I want to hit
    initProjectile(interceptor, interceptor_velocity, 0, 0);


    // Print stats for target
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

    double intercTimeToTarget;
    for(double angle=0.0; angle<90; angle+=.0001){ // For all firing angles, tiny incrememnt size because it's so fast
        // Update our interceptors angle
        interceptor->firingAngle = angle;

        // Get time to travel distance to intercept point so we can use it to verify x and y of collision
        intercTimeToTarget = timeGivenDistance(targetX, interceptor->firingAngle, interceptor->initialVelocity);
        if(intercTimeToTarget > total_travel_time){ // Skip angles where projectile would travel too long to get there
            continue;
        }
        // For this angle, get the interceptors y value when it passes through the target's x value
        // y = h + x * tan(α) - g * x² / (2 * V₀² * cos²(α))
        double intercYvalAtTargetX = predictedYValue(interceptor, targetX);

        // How far will our incerteptor have traveled in the time it takes to reach the target.
        // We do this because it may reach the target (x, y) coordinates, but at the wrong time
        double intercXvalAtTime = distanceGivenTime(intercTimeToTarget, interceptor->firingAngle, interceptor->initialVelocity);

        double differenceY = fabs(targetY - intercYvalAtTargetX);
        // Check to see if both y's are the same and if the time to target is positive, meaning the shot is possible. neg values are solutions but not in time
        if( (differenceY <= yToleranceToHit) && ((total_travel_time/2) - intercTimeToTarget) > 0.0){
            printf("-------Can hit target!------\n");
            printf("- Angle: %f\n- Time to Target: %f seconds\n- Launch after: %f seconds\n", angle, intercTimeToTarget, (total_travel_time/2) - intercTimeToTarget);
            printf("- Target(x, y): (%f, %f)\n- Interceptor(x, y): (%f, %f)\n- ",targetX, targetY, intercXvalAtTime, intercYvalAtTargetX);
            printEquation(interceptor);
            printf("----------------------------\n");
            angle += .1; // Increment angle by a solid amount to skip redundant firing solutions
        }
    }
    return 0;
}
