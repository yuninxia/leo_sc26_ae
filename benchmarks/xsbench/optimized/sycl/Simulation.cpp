#include "XSbench_header.h"

////////////////////////////////////////////////////////////////////////////////////
// LEO-OPTIMIZED SYCL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// Optimizations based on Leo's back-slicing root cause analysis on Intel PVC
// which identified memory-latency stalls as the dominant bottleneck (83.3% stall
// ratio). Top 3 stalls (47.9% combined) are in grid_search_nuclide binary search
// comparison and nuclide_grids energy loads.
//
// OPT 1: USM (Unified Shared Memory) instead of buffer/accessor
//   Raw device pointers avoid accessor indirection overhead.
//
// OPT 2: Energy field extraction (AoS → separate energy array)
//   Binary search only reads the 8-byte energy field but NuclideGridPoint is
//   48 bytes (6 doubles). A separate energy-only array gives 6x better cache
//   line utilization during the search loop. Full struct is only loaded after
//   the search locates the correct index for interpolation.
//
// OPT 3: Hybrid linear/binary search for small ranges
//   With hash grid mode, the binary search range (u_low..u_high) is typically
//   small. For ranges <= 8 elements, a linear scan avoids branch overhead
//   and is more predictable on PVC's wide SIMD.
//
// OPT 4: Work-group size tuning (64 for PVC)
//   PVC Xe-HPC execution units work best with sub-group-aligned work-groups.
////////////////////////////////////////////////////////////////////////////////////

using namespace sycl;

// Binary search on energy-only array (8 bytes per element vs 48 bytes with struct)
long grid_search_energy( long n, double quarry, const double * energy_array, long low, long high)
{
	long lowerLimit = low;
	long upperLimit = high;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );
		if( energy_array[examinationPoint] > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;
		length = upperLimit - lowerLimit;
	}

	return lowerLimit;
}

// Hybrid search: linear scan for small ranges, binary search for large
long grid_search_hybrid( long n, double quarry, const double * energy_array, long low, long high)
{
	long range = high - low;

	// For small ranges, linear scan avoids branch misprediction
	if( range <= 8 )
	{
		long best = low;
		for( long i = low + 1; i <= high; i++ )
		{
			if( energy_array[i] <= quarry )
				best = i;
			else
				break;
		}
		return best;
	}

	return grid_search_energy( n, quarry, energy_array, low, high );
}

long grid_search_usm( long n, double quarry, const double * A)
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

void calculate_micro_xs_usm( double p_energy, int nuc, long n_isotopes,
		long n_gridpoints,
		const double * egrid, const int * index_data,
		const NuclideGridPoint * nuclide_grids,
		const double * nuclide_grid_energy,
		long idx, double * xs_vector, int grid_type, int hash_bins )
{
	double f;
	NuclideGridPoint low, high;
	long low_idx, high_idx;

	if( grid_type == NUCLIDE )
	{
		long offset = nuc * n_gridpoints;
		// OPT 2: Search on energy-only array (8 bytes vs 48 bytes per element)
		idx = grid_search_energy( n_gridpoints, p_energy, nuclide_grid_energy, offset, offset + n_gridpoints-1);
		if( (idx - offset) == n_gridpoints - 1 )
			low_idx = idx - 1;
		else
			low_idx = idx;
	}
	else if( grid_type == UNIONIZED )
	{
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1;
		else
			low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc];
	}
	else // Hash grid
	{
		int u_low = index_data[idx * n_isotopes + nuc];

		int u_high;
		if( idx == hash_bins - 1 )
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

		long offset = nuc * n_gridpoints;
		// OPT 2: Use energy-only array for edge checks and search
		double e_low  = nuclide_grid_energy[offset + u_low];
		double e_high = nuclide_grid_energy[offset + u_high];
		long lower;
		if( p_energy <= e_low )
			lower = offset;
		else if( p_energy >= e_high )
			lower = offset + n_gridpoints - 1;
		else
		{
			// OPT 3: Hybrid search — linear for small ranges, binary for large
			lower = grid_search_hybrid( n_gridpoints, p_energy, nuclide_grid_energy, offset+u_low, offset+u_high);
		}

		if( (lower - offset) == n_gridpoints - 1 )
			low_idx = lower - 1;
		else
			low_idx = lower;
	}

	high_idx = low_idx + 1;
	// Only load full 48-byte struct AFTER search has identified the index
	low = nuclide_grids[low_idx];
	high = nuclide_grids[high_idx];

	f = (high.energy - p_energy) / (high.energy - low.energy);

	xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);
	xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);
	xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);
	xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);
	xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
}

void calculate_macro_xs_usm( double p_energy, int mat, long n_isotopes,
		long n_gridpoints, const int * num_nucs,
		const double * concs,
		const double * egrid, const int * index_data,
		const NuclideGridPoint * nuclide_grids,
		const double * nuclide_grid_energy,
		const int * mats,
		double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs )
{
	int p_nuc;
	long idx = -1;
	double conc;

	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	if( grid_type == UNIONIZED )
		idx = grid_search_usm( n_isotopes * n_gridpoints, p_energy, egrid);
	else if( grid_type == HASH )
	{
		double du = 1.0 / hash_bins;
		idx = p_energy / du;
	}

	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		double xs_vector[5];
		p_nuc = mats[mat*max_num_nucs + j];
		conc = concs[mat*max_num_nucs + j];
		calculate_micro_xs_usm( p_energy, p_nuc, n_isotopes,
				n_gridpoints, egrid, index_data,
				nuclide_grids, nuclide_grid_energy,
				idx, xs_vector, grid_type, hash_bins );
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype, double * kernel_init_time)
{
	if( mype == 0 ) printf("Allocating an additional %.1lf MB of memory for verification arrays...\n", in.lookups * sizeof(int) /1024.0/1024.0);
	int * verification_host = (int *) malloc(in.lookups * sizeof(int));

	double start = get_time();
	double stop;

	// Create SYCL queue
	queue sycl_q{default_selector_v};
	if(mype == 0 ) printf("Running on: %s\n", sycl_q.get_device().get_info<sycl::info::device::name>().c_str());
	if(mype == 0 ) printf("Initializing device buffers (USM) and JIT compiling kernel...\n");

	////////////////////////////////////////////////////////////////////////////////
	// Allocate device memory with USM
	////////////////////////////////////////////////////////////////////////////////
	int * d_num_nucs = sycl::malloc_device<int>(SD.length_num_nucs, sycl_q);
	double * d_concs = sycl::malloc_device<double>(SD.length_concs, sycl_q);
	int * d_mats = sycl::malloc_device<int>(SD.length_mats, sycl_q);
	NuclideGridPoint * d_nuclide_grid = sycl::malloc_device<NuclideGridPoint>(SD.length_nuclide_grid, sycl_q);
	// OPT 2: Energy-only array for cache-efficient binary search
	double * d_nuclide_grid_energy = sycl::malloc_device<double>(SD.length_nuclide_grid, sycl_q);
	int * d_verification = sycl::malloc_device<int>(in.lookups, sycl_q);

	// Handle potentially empty arrays (hash/nuclide grid modes don't use these)
	long ue_len = SD.length_unionized_energy_array > 0 ? SD.length_unionized_energy_array : 1;
	long ig_len = SD.length_index_grid > 0 ? SD.length_index_grid : 1;
	if( SD.length_unionized_energy_array == 0 )
		SD.unionized_energy_array = (double *) malloc(sizeof(double));
	if( SD.length_index_grid == 0 )
		SD.index_grid = (int *) malloc(sizeof(int));

	double * d_egrid = sycl::malloc_device<double>(ue_len, sycl_q);
	int * d_index_grid = sycl::malloc_device<int>(ig_len, sycl_q);

	// Copy data to device
	sycl_q.memcpy(d_num_nucs, SD.num_nucs, SD.length_num_nucs * sizeof(int));
	sycl_q.memcpy(d_concs, SD.concs, SD.length_concs * sizeof(double));
	sycl_q.memcpy(d_mats, SD.mats, SD.length_mats * sizeof(int));
	sycl_q.memcpy(d_nuclide_grid, SD.nuclide_grid, SD.length_nuclide_grid * sizeof(NuclideGridPoint));
	sycl_q.memcpy(d_nuclide_grid_energy, SD.nuclide_grid_energy, SD.length_nuclide_grid * sizeof(double));
	sycl_q.memcpy(d_egrid, SD.unionized_energy_array, ue_len * sizeof(double));
	sycl_q.memcpy(d_index_grid, SD.index_grid, ig_len * sizeof(int));
	sycl_q.wait();

	////////////////////////////////////////////////////////////////////////////////
	// OPT 4: Launch kernel with nd_range (work-group size = 64 for PVC)
	////////////////////////////////////////////////////////////////////////////////
	int wg_size = 64;
	int global_size = ((in.lookups + wg_size - 1) / wg_size) * wg_size;

	// Capture scalars for kernel lambda
	int lookups = in.lookups;
	long n_isotopes = in.n_isotopes;
	long n_gridpoints = in.n_gridpoints;
	int grid_type = in.grid_type;
	int hash_bins = in.hash_bins;
	int max_num_nucs = SD.max_num_nucs;

	sycl_q.submit([&](handler &cgh)
	{
		cgh.parallel_for<class kernel>(nd_range<1>(global_size, wg_size), [=](nd_item<1> item)
		{
			size_t i = item.get_global_id(0);

			if( i >= (size_t)lookups )
				return;

			uint64_t seed = STARTING_SEED;
			seed = fast_forward_LCG(seed, 2*i);

			double p_energy = LCG_random_double(&seed);
			int mat         = pick_mat(&seed);

			double macro_xs_vector[5] = {0};

			calculate_macro_xs_usm(
					p_energy, mat, n_isotopes, n_gridpoints,
					d_num_nucs, d_concs, d_egrid, d_index_grid,
					d_nuclide_grid, d_nuclide_grid_energy, d_mats,
					macro_xs_vector, grid_type, hash_bins, max_num_nucs );

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
			d_verification[i] = max_idx+1;
		});
	});

	sycl_q.wait();
	stop = get_time();
	if(mype==0) printf("Kernel initialization, compilation, and launch took %.2lf seconds.\n", stop-start);
	if(mype==0) printf("Beginning event based simulation...\n");

	// Copy verification results back to host
	sycl_q.memcpy(verification_host, d_verification, in.lookups * sizeof(int)).wait();

	// Free device memory
	sycl::free(d_num_nucs, sycl_q);
	sycl::free(d_concs, sycl_q);
	sycl::free(d_mats, sycl_q);
	sycl::free(d_nuclide_grid, sycl_q);
	sycl::free(d_nuclide_grid_energy, sycl_q);
	sycl::free(d_egrid, sycl_q);
	sycl::free(d_index_grid, sycl_q);
	sycl::free(d_verification, sycl_q);

	// Host reduces the verification array
	unsigned long long verification_scalar = 0;
	for( int i = 0; i < in.lookups; i++ )
		verification_scalar += verification_host[i];

	free(verification_host);
	return verification_scalar;
}


// Keep original template versions for compatibility (used by GridInit.cpp at host scope)
template <class T>
long grid_search( long n, double quarry, T A)
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

template <class Double_Type, class Int_Type, class NGP_Type>
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
		long n_gridpoints,
		Double_Type  egrid, Int_Type  index_data,
		NGP_Type  nuclide_grids,
		long idx, double *  xs_vector, int grid_type, int hash_bins ){
	double f;
	NuclideGridPoint low, high;
	long low_idx, high_idx;

	if( grid_type == NUCLIDE )
	{
		long offset = nuc * n_gridpoints;
		idx = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset, offset + n_gridpoints-1);
		if( idx == n_gridpoints - 1 )
			low_idx = idx - 1;
		else
			low_idx = idx;
	}
	else if( grid_type == UNIONIZED)
	{
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1;
		else
			low_idx = nuc*n_gridpoints + index_data[idx * n_isotopes + nuc];
	}
	else
	{
		int u_low = index_data[idx * n_isotopes + nuc];
		int u_high;
		if( idx == hash_bins - 1 )
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;
		double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
		double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
		long lower;
		if( p_energy <= e_low )
			lower = nuc*n_gridpoints;
		else if( p_energy >= e_high )
			lower = nuc*n_gridpoints + n_gridpoints - 1;
		else
		{
			long offset = nuc*n_gridpoints;
			lower = grid_search_nuclide( n_gridpoints, p_energy, nuclide_grids, offset+u_low, offset+u_high);
		}
		if( (lower % n_gridpoints) == n_gridpoints - 1 )
			low_idx = lower - 1;
		else
			low_idx = lower;
	}

	high_idx = low_idx + 1;
	low = nuclide_grids[low_idx];
	high = nuclide_grids[high_idx];
	f = (high.energy - p_energy) / (high.energy - low.energy);
	xs_vector[0] = high.total_xs - f * (high.total_xs - low.total_xs);
	xs_vector[1] = high.elastic_xs - f * (high.elastic_xs - low.elastic_xs);
	xs_vector[2] = high.absorbtion_xs - f * (high.absorbtion_xs - low.absorbtion_xs);
	xs_vector[3] = high.fission_xs - f * (high.fission_xs - low.fission_xs);
	xs_vector[4] = high.nu_fission_xs - f * (high.nu_fission_xs - low.nu_fission_xs);
}

template <class Double_Type, class Int_Type, class NGP_Type, class E_GRID_TYPE, class INDEX_TYPE>
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
		long n_gridpoints, Int_Type  num_nucs,
		Double_Type  concs,
		E_GRID_TYPE  egrid, INDEX_TYPE  index_data,
		NGP_Type  nuclide_grids,
		Int_Type  mats,
		double * macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
	int p_nuc;
	long idx = -1;
	double conc;

	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	if( grid_type == UNIONIZED )
		idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);
	else if( grid_type == HASH )
	{
		double du = 1.0 / hash_bins;
		idx = p_energy / du;
	}

	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		double xs_vector[5];
		p_nuc = mats[mat*max_num_nucs + j];
		conc = concs[mat*max_num_nucs + j];
		calculate_micro_xs( p_energy, p_nuc, n_isotopes,
				n_gridpoints, egrid, index_data,
				nuclide_grids, idx, xs_vector, grid_type, hash_bins );
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
}

int pick_mat( unsigned long * seed )
{
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

double LCG_random_double(uint64_t * seed)
{
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
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
