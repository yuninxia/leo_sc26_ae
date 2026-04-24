/*
 * SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//  This file is #include'd as the core of the computations on the GPU or CPU

#if 0
    // do nothing at all
#endif

#if 0
    // set the result to one in the kernel
    d_p1[i] = 1.;
#endif

#if 0
    // decrement the result by one in the kernel
    d_p1[i] = d_p1[i] -1.;
#endif

#if 0
      // use transcendental function in the kernel
#ifdef INFORTRAN
    do kk = 1, kernmax
       d_p1(i) = d_p1(i) + 1. + (sqrt( exp( log (d_l1(i)*d_l1(i)) ) + exp( log (d_r1(i)*d_r1(i)) ) ) ) / &
                                (sqrt( exp( log (d_l1(i)*d_r1(i)) ) + exp( log (d_r1(i)*d_l1(i)) ) ) )
    end do
#else
    for (int kk = 0 ; kk < kernmax ; kk++ ) {
      d_p1[i] = d_p1[i] + 1.+ (sqrt( exp( log (d_l1[i]*d_l1[i]) ) + exp( log (d_r1[i]*d_r1[i]) ) ) ) /
        ( sqrt (exp( log(d_l1[i]*d_r1[i]) ) + exp( log( (d_r1[i]*d_l1[i]) )) ) );
    }
#endif /* INFORTRAN */
#endif

#if 1
    // do a vector add in the kernel
#ifdef INFORTRAN
    do kk = 1, kernmax
       d_p1(i) = d_p1(i) + d_l1(nelements + 1 - kk) / real(kernmax, KIND=real64) + d_r1(kk) / real(kernmax, KIND=real64)
    end do
#else
    for (int kk = 0 ; kk < kernmax ; kk++ ) {
#if 0
      // The l1 and r1 arrays are initialized with element[i] = 1+i. The p1 array is zeroed.
      //   kernmax defaults to 2000.  nelements defaults to 40,000,000.

      // Original version, which reads d_l1 beyond its limit; works except for opencl
      //   which gets a memory access error.
      d_p1[i] = d_p1[i] + d_l1[nelements - kk] / (double)kernmax + d_r1[kk] / (double)kernmax;

      // Version that works, with all reads in bounds (and identical for all i)
      d_p1[i] = d_p1[i] + d_l1[nelements - 5] / (double)kernmax + d_r1[5] / (double)kernmax;

      // Version as originally intended, gets erroneous computation results everywhere
      d_p1[i] = d_p1[i] + d_l1[nelements - i-1] / (double)kernmax + d_r1[i] / (double)kernmax;

      // Version doing integer arithmetic, gets erroneous computation results everywhere
      d_p1[i] = d_p1[i] + (double) ( (int)d_l1[nelements-i-1] /kernmax  + (int)d_r1[i] /kernmax );

      // Version that works everywhere
      d_p1[i] = d_p1[i] + (double) ( (int)d_l1[nelements-i-1] + (int)d_r1[i] ) / (double)kernmax;

#endif
      d_p1[i] = d_p1[i] + (double) ( (int)d_l1[nelements-i-1] + (int)d_r1[i] ) / (double)kernmax;
#endif /* INFORTRAN */
}
#endif
