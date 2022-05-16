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
#include <math.h> // pow(), fabs()
#include <assert.h>
#include <stdlib.h>

#define SIZE	90.0
#define THREADS 400.0 // This is total thread count, not threads per block
#define BLOCKS  60.0 // Try and stick to even numbers

// __global__ Lets compiler know we can run this function on GPU.
__global__ void calculateFiringSolutionInAngleRange(double targetDistance, double projectileVelocity, double initialProjectileHeight){

    // --- Establish work start and stop
    // Need to get work from where the last blocks last thread's work stops
	double work_per_thread = SIZE/(THREADS*BLOCKS);
	// printf("work per thread %f\n", work_per_thread);

	// Split up work
	double work_start = work_per_thread * (THREADS * blockIdx.x + threadIdx.x);
	double work_stop;

	// Last thread pick up remainder
	if(threadIdx.x == (THREADS-1) && blockIdx.x == BLOCKS-1){
		work_stop = SIZE;
	}else{
		work_stop = work_per_thread * (THREADS * blockIdx.x + threadIdx.x + 1);
	}
    // --- Establish work start and stop

    // Initial variables
    double stepSize = 100000;
    double deltax = targetDistance/stepSize;
    
    // Within .5 centimeters on the x axis
    double xToleranceToHit = 0.005;

    // Within .5 centimeters on the y axis
    double yToleranceToHit = 0.005;

    // Increase launch angle by a very small number as to cover many possible trajectories
    for(double angle=work_start; angle<work_stop; angle += .005){
        // Variables for storing Riemann Sum values
        double x = 0;
        double projectileDistanceTraveled = 0;
        double angleInRadians = angle *  (M_PI / 180.0);


        while(1){
            // y = h + x * tan(α) - g * x² / (2 * V₀² * cos²(α)) // 4.9 because gravity is divided by 2
            double areaUnderSlice = initialProjectileHeight + x * tan(angleInRadians) - (9.8 * x * x / (2 * projectileVelocity * projectileVelocity * cos(angleInRadians) * cos(angleInRadians)));

            areaUnderSlice = areaUnderSlice * deltax;
            // If area under slice is negative, shot cannot reach or if we haven't found solution by the distance of the target
            if(areaUnderSlice < 0.0 || x > targetDistance){
                break;
            }

            // Total distance in x direction
            projectileDistanceTraveled += areaUnderSlice;

            // Check projectile elevation at given x, checking if less than zero, thus can't hit
            double projectileElevation = initialProjectileHeight + projectileDistanceTraveled * tan(angleInRadians) - (9.8 * projectileDistanceTraveled * projectileDistanceTraveled / (2 * projectileVelocity * projectileVelocity * cos(angleInRadians) * cos(angleInRadians)));
        
            // Y value is negative, can't hit target
            if(projectileElevation < 0.0){
                break;
            }

            // Candidate for hit, distance travelled by shot is almost equal to distance to target, now we need to check elevation
            if( fabs(targetDistance - projectileDistanceTraveled) <= xToleranceToHit){
                // We can check projectile elevation by plugging into f() function with our distance as the x value now
                if( projectileElevation <= yToleranceToHit){
                    double travelTime = (projectileDistanceTraveled / (cos(angleInRadians) * projectileVelocity));

                    printf("-- Hit Target! -- Projectile traveled %f meters in %f seconds with angle %f degrees.\n", projectileDistanceTraveled, travelTime, angle);
                    // printf("Projectile elevation: %f\n",projectileElevation);
                    break;
                }
            }

            x += deltax;
        }
        if(angle == 45.0){
            printf("Max projectile distance for %f = %f\n", angle, projectileDistanceTraveled);
        }
    }
}

int main(int argc, char* argv[]){
    if(argc < 3){
        printf("Usage: ./stationary_seq target_distance initial_projectile_velocity\n");
        exit(1);
    }

    double targetDistance = atof(argv[1]);
    double projectileVelocity = atof(argv[2]);
    double initialProjectileHeight = 0;

    // Not using gpu memory, should I be? // cudaMallocManaged(&targetDistance, 2 * sizeof(double));

    calculateFiringSolutionInAngleRange <<<BLOCKS, THREADS>>>(targetDistance, projectileVelocity, initialProjectileHeight);

    // Like join? Or barrier?
	cudaDeviceSynchronize();
    
    return 0;
}
