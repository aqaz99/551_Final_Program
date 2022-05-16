
// https://www.omnicalculator.com/physics/range-projectile-motion

#include <stdio.h>
#include <math.h> // pow()
#include <assert.h>
#include <stdlib.h>

typedef struct {
  double xPosition, yPosition, initialVelocity, initialHeight, firingAngle;
} ProjectileClass;

double predictedYValue(ProjectileClass* projectile, double x);
void initProjectile(ProjectileClass* projectile, double x, double y, double initialVelocity, double initialHeight, double firingAngle);
double distanceGivenTime(double time, double angle, double projectileVelocity);
double projectileTravelTime(double distance, double angle, double projectileVelocity);
double calculateTotalDistance(ProjectileClass* projectile);

int main(int argc, char* argv[]){

    // Within 10 micrometers on the x axis
    double xToleranceToHit = 0.0005;

    // Within 10 centimeters on the y axis
    double yToleranceToHit = 0.0005;

    // Init moving target with initial velocity of 35 and firing angle of 45
    ProjectileClass* target = malloc(sizeof(ProjectileClass)); // Object I want to hit
    initProjectile(target, 0, 0, 35, 0, 45);

    // Print stats for target
    double total_distance_traveled = calculateTotalDistance(target);
    printf("Total distance traveled: %f\n", total_distance_traveled);
    double total_travel_time = projectileTravelTime(total_distance_traveled, target->firingAngle, target->initialVelocity);
    printf("Total travel time: %f\n", total_travel_time);



    // Init interceptor with initial velocity of 45 and firing angle of 0
    ProjectileClass* interceptor = malloc(sizeof(ProjectileClass)); // Object I want to hit
    initProjectile(interceptor, 0, 0, 205, 0, 0);

    // double time = 3.0;
    // double distance = distanceGivenTime(time, target->firingAngle, target->initialVelocity);

    // printf("Target distance at %f: %f\n", time, distance);

    // predictedYValue(target, distance);
    // getPredictedTargetYValue(interceptor, x);

    // Now we want to check at which firing angle we can intercept the projectile
    // We will need time, x, and y to be equal.
    for(double angle=0.0; angle<90; angle+=.001){ // For all firing angles
        interceptor->firingAngle = angle;
        // Compare target (x,y) with interceptor (x,y) at each time step, if they are equal we have a solution
        for(double airTime=1.0; airTime < total_travel_time; airTime+= .01){
            // Get X values of target and interceptor
            double targetX = distanceGivenTime(airTime, target->firingAngle, target->initialVelocity);
            double intercX = distanceGivenTime(airTime, interceptor->firingAngle, interceptor->initialVelocity);

            // Get Y valus
            double targetY = predictedYValue(target, targetX);
            double intercY = predictedYValue(target, intercX);

            double differenceX = fabs(targetX - intercX);
            double differenceY = fabs(targetY - intercY);

            // printf("%f and %f\n", differenceX, differenceY);
            if( (differenceX <= xToleranceToHit) && (differenceY <= yToleranceToHit)){
                printf("-------Can hit target!------\n");
                printf("Angle: %f\nTime Step: %f\n", angle, airTime);
                printf("Target(x, y): (%f, %f)\nInterceptor(x, y): (%f, %f)\n",targetX, targetY, intercX, intercY);
                printf("----------------------------\n");
                break;
            }
        }
    }

    return 0;
}


// Equation taken from: https://www.omnicalculator.com/physics/trajectory-projectile-motion
double predictedYValue(ProjectileClass* projectile, double x){
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    double y;
    double underTheDivision = (2 * projectile->initialVelocity * projectile->initialVelocity * cos(angleInRadians) * cos(angleInRadians));

    // printf("Under the division (IN METERS): %f\n", underTheDivision);
    
    // y = h + x * tan(α) - g * x² / (2 * V₀² * cos²(α)) // 4.9 because gravity is divided by 2
    y = projectile->initialHeight + x * tan(angleInRadians) - (9.8 * x * x / underTheDivision);

    // printf("Projectiles y value at x=%f is %f\n",x, y);
    return y;
}

void initProjectile(ProjectileClass* projectile, double x, double y, double initialVelocity, double initialHeight, double firingAngle){
    projectile->xPosition = x;
    projectile->yPosition = y;
    projectile->initialVelocity = initialVelocity;
    projectile->initialHeight = initialHeight;
    projectile->firingAngle = firingAngle;
}

double distanceGivenTime(double time, double angle, double projectileVelocity){
    // distance = rate * time
    // Total distance traveled -> distance(x) = time * rate, where rate is cos(theta) * velocity
    return (time * cos(angle) * projectileVelocity);
}

double projectileTravelTime(double distance, double angle, double projectileVelocity){
    double angleInRadians = angle *  (M_PI / 180.0);
    // distance = rate * time
    // Total travel time -> Time = distance(x) / rate(cos(theta) * velocity)
    return (distance / (cos(angleInRadians) * projectileVelocity));
}

// https://www.omnicalculator.com/physics/range-projectile-motion
// This equation allows us to calculate the range of a projectile with height >= 0
// R = Vx * [Vy + √(Vy² + 2 * g * h)] / g
double calculateTotalDistance(ProjectileClass* projectile){

    // Get velocity in the x and y directions
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    double Vx = projectile->initialVelocity * cos(angleInRadians);
    double Vy = projectile->initialVelocity * sin(angleInRadians);
    
    return (Vx * (Vy + sqrt(Vy * Vy + 2 * 9.8 * projectile->initialHeight)) / 9.8);
}