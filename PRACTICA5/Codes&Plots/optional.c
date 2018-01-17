double relax_gauss (double *u, unsigned sizex, unsigned sizey){
	double unew, diff, sum=0.0;
	int howmany=omp_get_max_threads();
	int howmanyAux = howmany;
	char dep[howmany][howmanyAux];
	omp_lock_t lock;
	omp_init_lock(&lock);
	#pragma omp parallel
	#pragma omp single
	for (int blockid = 0; blockid < howmany; ++blockid) {
		 int i_start = lowerb(blockid, howmany, sizex);
		 int i_end = upperb(blockid, howmany, sizex);
		 for (int z = 0; z < howmanyAux; z++) {
			 int j_start = lowerb(z, howmanyAux, sizey);
			 int j_end = upperb(z,howmanyAux, sizey);
			 #pragma omp task firstprivate (j_start,j_end, i_start, i_end)
			 depend(in: dep[max(blockid-1,0)][z], dep[blockid][max(0,z-1)])
			 depend (out: dep[blockid][z]) private(diff,unew)
			 {
	 			 double sum2 = 0.0;
					​for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
	 					for (int j = max(1, j_start); j<= min(j_end, sizey-2); j++) {
							 unew= 0.25 * ( u[ i*sizey + (j-1) ]+ // left
											u[ i*sizey + (j+1) ]+ // right
											 u[ (i-1)*sizey + j ]+ // top
											 u[ (i+1)*sizey + j ]); // bottom
							 diff = unew - u[i*sizey+ j];
							 sum2 += diff * diff;
							 u[i*sizey+j]=unew;
						}
					}
				 omp_set_lock(&lock);
				 sum += sum2;
				 omp_unset_lock(&lock);
 			}
 		}
	}
	omp_destroy_lock(&lock);
	return sum;
}