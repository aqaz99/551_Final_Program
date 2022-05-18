#include <stdio.h>
#include <math.h> // pow()

typedef struct {
  double initialVelocity, initialHeight, firingAngle;
} ProjectileClass;

// Equation taken from: https://www.omnicalculator.com/physics/trajectory-projectile-motion
double predictedYValue(ProjectileClass* projectile, double x){
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    double y;
    double underTheDivision = (2 * projectile->initialVelocity * projectile->initialVelocity * cos(angleInRadians) * cos(angleInRadians));

    // printf("Under the division (IN METERS): %f\n", underTheDivision);
    
    // y = h + x * tan(α) - g * x² / (2 * V₀² * cos²(α)) // 4.9 because gravity is divided by 2
    y = projectile->initialHeight + x * tan(angleInRadians) - (9.8 * x * x / underTheDivision);

    return y;
}

void initProjectile(ProjectileClass* projectile, double initialVelocity, double target_initial_height, double firingAngle){
    projectile->initialVelocity = initialVelocity;
    projectile->initialHeight = target_initial_height;
    projectile->firingAngle = firingAngle;
}

double distanceGivenTime(double time, double angle, double projectileVelocity){
    double angleInRadians = angle *  (M_PI / 180.0);
    // distance = rate * time
    // Total distance traveled -> distance(x) = time * rate, where rate is cos(theta) * velocity
    return (time * cos(angleInRadians) * projectileVelocity);
}

double timeGivenDistance(double distance, double angle, double projectileVelocity){
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

void printEquation(ProjectileClass* projectile){
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    double underTheDivision = (2 * projectile->initialVelocity * projectile->initialVelocity * cos(angleInRadians) * cos(angleInRadians));
    printf("Function equation F(x) = %f + %fx - (9.8x^2) / %f\n",projectile->initialHeight, tan(angleInRadians), underTheDivision);
}

