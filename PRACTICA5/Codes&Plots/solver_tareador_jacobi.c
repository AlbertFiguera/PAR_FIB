
#include "heat.h"
#include <stdio.h>
#include <tareador.h>

/*
 * Function to copy one matrix into another
 */

void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey)
{
    for (int i=1; i<= sizex-2; i++)
        for (int j=1; j<= sizey-2; j++) 
            v[ i*sizey+j ] = u[ i*sizey+j ];
}

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
  
    int howmany=1;
    for (int blockid = 0; blockid < howmany; ++blockid) {
      int i_start = lowerb(blockid, howmany, sizex);
      int i_end = upperb(blockid, howmany, sizex);
      for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
        for (int j=1; j<= sizey-2; j++) {
	tareador_start_task("Jacobi");
	tareador_disable_object(&sum);     
	utmp[i*sizey+j]= 0.25 * ( u[ i*sizey     + (j-1) ]+  // left
	                               u[ i*sizey     + (j+1) ]+  // right
				       u[ (i-1)*sizey + j     ]+  // top
				       u[ (i+1)*sizey + j     ]); // bottom
	     diff = utmp[i*sizey+j] - u[i*sizey + j];
	
	     sum += diff * diff;
	tareador_enable_object(&sum);
	tareador_end_task("Jacobi");
	 }
      }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;

    int howmany=1;
    for (int blockid = 0; blockid < howmany; ++blockid) {
      int i_start = lowerb(blockid, howmany, sizex);
      int i_end = upperb(blockid, howmany, sizex);
      for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
        for (int j=1; j<= sizey-2; j++) {
	tareador_start_task("Gauss");
	tareador_disable_object(&sum);  
	    unew= 0.25 * ( u[ i*sizey	+ (j-1) ]+  // left
			   u[ i*sizey	+ (j+1) ]+  // right
			   u[ (i-1)*sizey	+ j     ]+  // top
			   u[ (i+1)*sizey	+ j     ]); // bottom
	    diff = unew - u[i*sizey+ j];
	    sum += diff * diff;
	tareador_enable_object(&sum); 
	    u[i*sizey+j]=unew;
	tareador_end_task("Gauss");        
	}
      }
    }

    return sum;
}
