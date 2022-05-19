
// https://www.omnicalculator.com/physics/range-projectile-motion

#include <stdio.h>
#include <math.h> // pow()
#include <assert.h>
#include <stdlib.h>

#define SIZE	90.0
#define THREADS 100.0 // This is total thread count, not threads per block
#define BLOCKS  50.0 // Try and stick to even numbers

typedef struct {
  double xPosition, yPosition, initialVelocity, initialHeight, firingAngle;
} ProjectileClass;

// __device__ means the function is called by GPU and is supposed to be executed on GPU
__device__ __host__ double predictedYValue(ProjectileClass* projectile, double x);
// __host__ means the functions is called by the CPU and is supposed to be executed on the CPU
__host__ void initProjectile(ProjectileClass* projectile, double initialVelocity, double initialHeight, double firingAngle);
__device__ __host__ double distanceGivenTime(double time, double angle, double projectileVelocity);
__host__ __device__ double timeGivenDistance(double distance, double angle, double projectileVelocity);
__host__ double calculateTotalDistance(ProjectileClass* projectile);
__host__ __device__ void printEquation(ProjectileClass* projectile);


// __global__ means the function is called by CPU and suppose to be executed on GPU
__global__ void calculateFiringSolutionInAngleRange(ProjectileClass* target, 
                                                    ProjectileClass** interceptorArray, 
                                                    double total_travel_time,
                                                    double targetX,
                                                    double targetY,
                                                    int* foundSolution){


    // printf("thread[%d] computing from %f to %f\n", ((blockIdx.x * blockDim.x) + threadIdx.x), work_start, work_stop);
    
    ProjectileClass* myInterceptor = interceptorArray[((blockIdx.x * blockDim.x) + threadIdx.x)];
    // Within .05 meters or ~2 inches
    double xToleranceToHit = 0.05;

    // Within .05 meters or ~2 inches
    double yToleranceToHit = 0.05;

    // Lets say we want to hit our target at (travel time)/2 
    // 1. Find the x and y location of our target at that time
    // 2. Find an angle that an interceptor could be fired at that passes through that (x, y)
    // 3. Determine the time it would take for the interceptor to get to that location
    // 4. If the interceptor can reach that location in time, display when the interceptor needs to fire to hit the target

    // Initial Riemann sum variables
    double stepSize = 15000;
    double deltax = targetX/stepSize;

    // Variables for storing Riemann Sum values
    double x = 0;
    double projectileDistanceTraveled = 0;
    while(*foundSolution != 1){
        // Should this x be dist traveled?
        double areaUnderSlice = predictedYValue(myInterceptor, x);
        areaUnderSlice = areaUnderSlice * deltax;
        // printf("area under slice: %f\n", areaUnderSlice);
        // If area under slice is negative, shot cannot reach or if we haven't found solution by the distance of the target
        if(areaUnderSlice < 0.0){
            break;
        }

        // Total distance in x direction
        projectileDistanceTraveled += areaUnderSlice;
        double projectileElevation = predictedYValue(myInterceptor, projectileDistanceTraveled);

        // Get time it will take our projectile to hit the target
        double intercTimeToTarget = timeGivenDistance(targetX, myInterceptor->firingAngle, myInterceptor->initialVelocity);
        if(intercTimeToTarget > total_travel_time){ // Skip angles where projectile would travel too long to get there
            break;
        }

        // Y value is negative, can't hit target
        if(projectileElevation < 0.0){
            break;
        }
        // Candidate for hit, distance travelled by shot is almost equal to distance to target, now we need to check elevation
            // printf("- Target(x, y): (%f, %f)\n- Interceptor(x, y): (%f, %f)\n- ",targetX, targetY, projectileDistanceTraveled, projectileElevation);
        if( fabs(targetX - projectileDistanceTraveled) <= xToleranceToHit){

            // printf("%f - %f = %f , intercTimeToTarget: %f\n",projectileElevation, targetY, abs(projectileElevation - targetY), intercTimeToTarget);
            // Check to see if both y's are the same and if the time to target is positive, meaning the shot is possible. neg values are solutions but not in time
            if( fabs(projectileElevation - targetY) <= yToleranceToHit && ((total_travel_time/2) - intercTimeToTarget) > 0.0){
                *foundSolution = 1;
                printf("-------Can hit target!------\n");
                printf("- Angle: %f\n- Time to Target: %f seconds\n- Launch after: %f seconds\n", myInterceptor->firingAngle, intercTimeToTarget, (total_travel_time/2) - intercTimeToTarget);
                printf("- Target(x, y): (%f, %f)\n- Interceptor(x, y): (%f, %f)\n- ",targetX, targetY, projectileDistanceTraveled, projectileElevation);
                printEquation(myInterceptor);
                printf("----------------------------\n");
                break;
            }
        }
        x += deltax;
    }
    // printf("Total distance for interceptor: %f\n",projectileDistanceTraveled);
}

int main(int argc, char* argv[]){
    // CL Arguments
    if(argc !=  5){
        printf("Usage: ./moving_cuda TARGET_velocity TARGET_initial_height TARGET_firing_angle INTERCEPTOR_velocity\n");
        exit(1);
    }

    double target_velocity = atof(argv[1]);
    double target_initial_height = atof(argv[2]);
    double target_firing_angle = atof(argv[3]);
    double interceptor_velocity = atof(argv[4]);

    // Init moving target with initial velocity of 35 and firing angle of 45
    ProjectileClass* target;
    cudaMallocManaged(&target, SIZE * sizeof(ProjectileClass));
    initProjectile(target, target_velocity, target_initial_height, target_firing_angle);
    printEquation(target);

    // Need to allocate array of interceptors that start at angles 0.00 -> 90.00 in increments of SIZE/(THREADS*BLOCKS)
    // You cannot allocate memory on the GPU after executing the function call so we do it on the cpu before the call
    ProjectileClass** interceptorArray;
    cudaMallocManaged(&interceptorArray, THREADS * BLOCKS * sizeof(ProjectileClass));
    double work_per_thread = SIZE/(THREADS*BLOCKS);

	// Split up work
    for(int i=0; i<THREADS * BLOCKS; i++){
        // Each thread gets their own interceptor
        ProjectileClass* interceptor;
        cudaMallocManaged(&interceptor, sizeof(ProjectileClass));
	    double work_start = work_per_thread * i;
        initProjectile(interceptor, interceptor_velocity, 0, work_start);
        interceptorArray[i] = interceptor;
    }

    //print info about target projectile
    double total_distance_traveled = calculateTotalDistance(target);
    printf("Target final x position: %f\n", total_distance_traveled);
    double total_travel_time = timeGivenDistance(total_distance_traveled, target->firingAngle, target->initialVelocity);
    printf("Total travel time: %f\n", total_travel_time);


    // 1. Get (x, y) at total time / 2 
    printf("Attempting to hit target at %f\n", total_travel_time/2);
    double targetX = distanceGivenTime(total_travel_time/2, target->firingAngle, target->initialVelocity);
    double targetY = predictedYValue(target, targetX);
    printf("Targets position at %f seconds is (%f, %f)\n",total_travel_time/2, targetX, targetY);

    // Shared var between threads to quit once a solution is found
    int* foundSolution;
    foundSolution = 0;
    cudaMallocManaged(&foundSolution, sizeof(int));

    // Timing 
    float elapsedTime=0;
    cudaEvent_t start, finish;

    // Init timer events
    cudaEventCreate(&start);
    cudaEventCreate(&finish);

    // Start timing
    cudaEventRecord(start, 0);

    // Issue here is that I am executing code on the gpu, but calling functions on the cpu, need to remedy
    calculateFiringSolutionInAngleRange <<<BLOCKS, THREADS>>>(target, interceptorArray, total_travel_time, targetX, targetY, foundSolution);

    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);

    cudaEventElapsedTime(&elapsedTime, start, finish);

    cudaEventDestroy(start);
    cudaEventDestroy(finish);

    printf("Total elapsed time in gpu was %.2f seconds\n", elapsedTime/1000);

    // Like join from pthreads
	cudaDeviceSynchronize();
    return 0;
}


// Equation taken from: https://www.omnicalculator.com/physics/trajectory-projectile-motion
__device__ double predictedYValue(ProjectileClass* projectile, double x){
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    // printf("x: %f tan(angleinrads): %f\n", x,tan(angleInRadians));
    double y;
    double underTheDivision = (2 * projectile->initialVelocity * projectile->initialVelocity * cos(angleInRadians) * cos(angleInRadians));

    // printf("Under the division (IN METERS): %f\n", underTheDivision);
    
    // y = h + x * tan(α) - g * x² / (2 * V₀² * cos²(α)) // 4.9 because gravity is divided by 2
    y = projectile->initialHeight + x * tan(angleInRadians) - (9.8 * x * x / underTheDivision);
    // printf("Y: %f\n", y);
    return y;
}

__host__ void initProjectile(ProjectileClass* projectile, double initialVelocity, double initialHeight, double firingAngle){
    projectile->initialVelocity = initialVelocity;
    projectile->initialHeight = initialHeight;
    projectile->firingAngle = firingAngle;
}

__device__ double distanceGivenTime(double time, double angle, double projectileVelocity){
    double angleInRadians = angle *  (M_PI / 180.0);
    // distance = rate * time
    // Total distance traveled -> distance(x) = time * rate, where rate is cos(theta) * velocity
    return (time * cos(angleInRadians) * projectileVelocity);
}

__host__ double timeGivenDistance(double distance, double angle, double projectileVelocity){
    double angleInRadians = angle *  (M_PI / 180.0);
    // distance = rate * time
    // Total travel time -> Time = distance(x) / rate(cos(theta) * velocity)
    return (distance / (cos(angleInRadians) * projectileVelocity));
}

// https://www.omnicalculator.com/physics/range-projectile-motion
// This equation allows us to calculate the range of a projectile with height >= 0
// R = Vx * [Vy + √(Vy² + 2 * g * h)] / g
__host__ double calculateTotalDistance(ProjectileClass* projectile){

    // Get velocity in the x and y directions
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    double Vx = projectile->initialVelocity * cos(angleInRadians);
    double Vy = projectile->initialVelocity * sin(angleInRadians);
    
    return (Vx * (Vy + sqrt(Vy * Vy + 2 * 9.8 * projectile->initialHeight)) / 9.8);
}

__host__ __device__ void printEquation(ProjectileClass* projectile){
    double angleInRadians = projectile->firingAngle *  (M_PI / 180.0);
    double underTheDivision = (2 * projectile->initialVelocity * projectile->initialVelocity * cos(angleInRadians) * cos(angleInRadians));
    printf("Function equation F(x) = %f + %fx - (9.8x^2) / %f\n",projectile->initialHeight, tan(angleInRadians), underTheDivision);
}