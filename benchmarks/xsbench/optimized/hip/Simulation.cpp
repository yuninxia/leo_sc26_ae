#include "hip/hip_runtime.h"
#include "XSbench_header.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// port of the original CPU OpenMP code to CUDA with few significant changes or
// optimizations made. Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, XSBench will only run the baseline implementation. Optimized variants
// must be specifically selected using the "-k <optimized variant ID>" command
// line argument.
////////////////////////////////////////////////////////////////////////////////////

unsigned long long run_event_based_simulation_baseline(Inputs in, SimulationData GSD, int mype)
{
	////////////////////////////////////////////////////////////////////////////////
	// Configure & Launch Simulation Kernel
	////////////////////////////////////////////////////////////////////////////////
	if( mype == 0)	printf("Running baseline event-based simulation...\n");

	// OPT: Use smaller workgroup (64 threads = 1 wave on AMD) to improve
	// occupancy. With 256 threads (4 waves/workgroup) and 56 VGPRs/wave,
	// only 2 workgroups fit per SIMD → 8 waves/CU → 20% occupancy.
	// With 64 threads (1 wave/workgroup), up to 9 workgroups fit per
	// SIMD → 36 waves/CU → 90% occupancy. More active waves means
	// better latency hiding for the memory-bound random lookups.
	int nthreads = 64;
	int nblocks = ceil( (double) in.lookups / (double) nthreads);

	// Kernel-only timing via hipEvent
	hipEvent_t kstart, kstop;
	hipEventCreate(&kstart);
	hipEventCreate(&kstop);
	hipEventRecord(kstart, 0);

	hipLaunchKernelGGL(xs_lookup_kernel_baseline, dim3(nblocks), dim3(nthreads), 0, 0,  in, GSD );
	gpuErrchk( hipPeekAtLastError() );

	hipEventRecord(kstop, 0);
	hipEventSynchronize(kstop);
	float kernel_ms = 0;
	hipEventElapsedTime(&kernel_ms, kstart, kstop);
	if( mype == 0 ) printf("Kernel time: %.6f seconds\n", kernel_ms / 1000.0);
	hipEventDestroy(kstart);
	hipEventDestroy(kstop);

	////////////////////////////////////////////////////////////////////////////////
	// Reduce Verification Results
	////////////////////////////////////////////////////////////////////////////////
	if( mype == 0)	printf("Reducing verification results...\n");


  size_t sz = in.lookups * sizeof(unsigned long);
  unsigned long * v = (unsigned long *) malloc(sz);
  gpuErrchk( hipMemcpy(v, GSD.verification, sz, hipMemcpyDeviceToHost) );

  unsigned long verification_scalar = 0;
  for( int i =0; i < in.lookups; i++ )
    verification_scalar += v[i];

	return verification_scalar;
}

__global__ void xs_lookup_kernel_baseline(Inputs in, SimulationData GSD )
{
	// The lookup ID. Used to set the seed, and to store the verification value
	const int i = blockIdx.x *blockDim.x + threadIdx.x;

	if( i >= in.lookups )
		return;

	// Set the initial seed value
	uint64_t seed = STARTING_SEED;

	// Forward seed to lookup index (we need 2 samples per lookup)
	seed = fast_forward_LCG(seed, 2*i);

	// Randomly pick an energy and material for the particle
	double p_energy = LCG_random_double(&seed);
	int mat         = pick_mat(&seed);

	double macro_xs_vector[5] = {0};

	// Perform macroscopic Cross Section Lookup
	calculate_macro_xs(
			p_energy,        // Sampled neutron energy (in lethargy)
			mat,             // Sampled material type index neutron is in
			in.n_isotopes,   // Total number of isotopes in simulation
			in.n_gridpoints, // Number of gridpoints per isotope in simulation
			GSD.num_nucs,     // 1-D array with number of nuclides per material
			GSD.concs,        // Flattened 2-D array with concentration of each nuclide in each material
			GSD.unionized_energy_array, // 1-D Unionized energy array
			GSD.index_grid,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
			GSD.nuclide_grid, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
			GSD.nuclide_grid_energy, // OPT: Energy-only array for cache-efficient binary search
			GSD.mats,         // Flattened 2-D array with nuclide indices defining composition of each type of material
			macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
			in.grid_type,    // Lookup type (nuclide, hash, or unionized)
			in.hash_bins,    // Number of hash bins used (if using hash lookup type)
			GSD.max_num_nucs  // Maximum number of nuclides present in any material
			);

	// For verification, and to prevent the compiler from optimizing
	// all work out, we interrogate the returned macro_xs_vector array
	// to find its maximum value index, then increment the verification
	// value by that index. In this implementation, we have each thread
	// write to its thread_id index in an array, which we will reduce
	// with a thrust reduction kernel after the main simulation kernel.
	double max = -1.0;
	int max_idx = 0;
	for(int j = 0; j < 5; j++ )
	{
		if( macro_xs_vector[j] > max )
		{
			max = macro_xs_vector[j];
			max_idx = j;
		}
	}
	GSD.verification[i] = max_idx+1;
}

// Calculates the microscopic cross section for a given nuclide & energy
__device__ void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           double * __restrict__ egrid, int * __restrict__ index_data,
                           NuclideGridPoint * __restrict__ nuclide_grids,
                           double * __restrict__ nuclide_grid_energy,
                           long idx, double * __restrict__ xs_vector, int grid_type, int hash_bins ){
	// Variables
	double f;
	NuclideGridPoint * low, * high;
	long low_idx;

	// Precompute base offset for this nuclide's grid (avoids repeated multiply)
	long nuc_offset = nuc * n_gridpoints;

	// If using only the nuclide grid, we must perform a binary search
	// to find the energy location in this particular nuclide's grid.
	if( grid_type == NUCLIDE )
	{
		// OPT: Search on energy-only array (8 bytes vs 48 bytes per element)
		idx = grid_search_energy( n_gridpoints, p_energy, nuclide_grid_energy, nuc_offset, nuc_offset + n_gridpoints-1);

		// pull ptr from nuclide grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( (idx - nuc_offset) == n_gridpoints - 1 )
			low_idx = idx - 1;
		else
			low_idx = idx;
	}
	else if( grid_type == UNIONIZED) // Unionized Energy Grid - we already know the index, no binary search needed.
	{
		// pull ptr from energy grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low_idx = nuc_offset + index_data[idx * n_isotopes + nuc] - 1;
		else
			low_idx = nuc_offset + index_data[idx * n_isotopes + nuc];
	}
	else // Hash grid
	{
		// load lower bounding index
		int u_low = index_data[idx * n_isotopes + nuc];

		// Determine higher bounding index
		int u_high;
		if( idx == hash_bins - 1 )
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

		// OPT: Fast path for narrow hash bins (range <= 1 element).
		// With 10000 hash bins and 11303 gridpoints, average bin width
		// is ~1.1, so >90% of lookups land here. This eliminates 2
		// dependent global loads (e_low, e_high) for the common case.
		int range = u_high - u_low;
		long lower;
		if( range <= 1 )
		{
			lower = nuc_offset + u_low;
		}
		else if( range <= 8 )
		{
			// Linear scan on energy-only array for small ranges
			double * __restrict__ eng = nuclide_grid_energy + nuc_offset;
			lower = nuc_offset + u_low;
			for( int k = u_low + 1; k <= u_high; k++ )
			{
				if( eng[k] > p_energy )
					break;
				lower = nuc_offset + k;
			}
		}
		else
		{
			// OPT: Use energy-only array for edge checks and search
			double e_low  = nuclide_grid_energy[nuc_offset + u_low];
			double e_high = nuclide_grid_energy[nuc_offset + u_high];
			if( p_energy <= e_low )
				lower = nuc_offset;
			else if( p_energy >= e_high )
				lower = nuc_offset + n_gridpoints - 1;
			else
				lower = grid_search_energy( n_gridpoints, p_energy, nuclide_grid_energy, nuc_offset+u_low, nuc_offset+u_high);
		}

		if( (lower - nuc_offset) == n_gridpoints - 1 )
			low_idx = lower - 1;
		else
			low_idx = lower;
	}

	// Only load full 48-byte struct AFTER search has identified the index
	low = &nuclide_grids[low_idx];
	high = low + 1;

	// calculate the re-useable interpolation factor
	f = (high->energy - p_energy) / (high->energy - low->energy);

	// Total XS
	xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);

	// Elastic XS
	xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);

	// Absorbtion XS
	xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);

	// Fission XS
	xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);

	// Nu Fission XS
	xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
}

// Calculates macroscopic cross section based on a given material & energy
__device__ void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int * __restrict__ num_nucs,
                         double * __restrict__ concs,
                         double * __restrict__ egrid, int * __restrict__ index_data,
                         NuclideGridPoint * __restrict__ nuclide_grids,
                         double * __restrict__ nuclide_grid_energy,
                         int * __restrict__ mats,
                         double * __restrict__ macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
	int p_nuc; // the nuclide we are looking up
	long idx = -1;
	double conc; // the concentration of the nuclide in the material

	// cleans out macro_xs_vector
	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	// If we are using the unionized energy grid (UEG), we only
	// need to perform 1 binary search per macroscopic lookup.
	// If we are using the nuclide grid search, it will have to be
	// done inside of the "calculate_micro_xs" function for each different
	// nuclide in the material.
	if( grid_type == UNIONIZED )
		idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);
	else if( grid_type == HASH )
	{
		// OPT: Multiplication instead of division for hash bin index
		idx = (long)(p_energy * (double)hash_bins);
		if (idx >= hash_bins) idx = hash_bins - 1;
	}

	// Precompute material offset for the nuclide/concentration arrays
	int mat_offset = mat * max_num_nucs;

	// Once we find the pointer array on the UEG, we can pull the data
	// from the respective nuclide grids, as well as the nuclide
	// concentration data for the material
	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		double xs_vector[5];
		p_nuc = mats[mat_offset + j];
		conc = concs[mat_offset + j];
		calculate_micro_xs( p_energy, p_nuc, n_isotopes,
		                    n_gridpoints, egrid, index_data,
		                    nuclide_grids, nuclide_grid_energy,
		                    idx, xs_vector, grid_type, hash_bins );
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}


// binary search for energy on unionized energy grid
// returns lower index
__device__ long grid_search( long n, double quarry, double * __restrict__ A)
{
	long lowerLimit = 0;
	long upperLimit = n-1;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );

		if( A[examinationPoint] > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;

		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
}

// binary search for energy on nuclide energy grid
__host__ __device__ long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high)
{
	long lowerLimit = low;
	long upperLimit = high;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );

		if( A[examinationPoint].energy > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;

		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
}

// OPT: Binary search on energy-only array (8 bytes per element vs 48 bytes with struct)
// This gives 6x better cache line utilization during the search loop.
__device__ long grid_search_energy( long n, double quarry, double * __restrict__ A, long low, long high)
{
	long lowerLimit = low;
	long upperLimit = high;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );

		if( A[examinationPoint] > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;

		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
}

// picks a material based on a probabilistic distribution
__device__ int pick_mat( uint64_t * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	// Also could be argued that doing fractions by weight would be
	// a better approximation, but volume does a good enough job for now.

	double dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies

	double roll = LCG_random_double(seed);

	// makes a pick based on the distro
	for( int i = 0; i < 12; i++ )
	{
		double running = 0;
		for( int j = i; j > 0; j-- )
			running += dist[j];
		if( roll < running )
			return i;
	}

	return 0;
}

__host__ __device__ double LCG_random_double(uint64_t * seed)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}

__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0)
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;
}
