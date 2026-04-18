#include "CollisionEvent.hh"
#include "MC_Particle.hh"
#include "NuclearData.hh"
#include "DirectionCosine.hh"
#include "MonteCarlo.hh"
#include "MC_Cell_State.hh"
#include "MaterialDatabase.hh"
#include "MacroscopicCrossSection.hh"
#include "MC_Base_Particle.hh"
#include "ParticleVaultContainer.hh"
#include "PhysicalConstants.hh"
#include "DeclareMacro.hh"
#include "QS_atomics.hh"

#define MAX_PRODUCTION_SIZE 4

//----------------------------------------------------------------------------------------------------------------------
//  Leo-guided optimization of CollisionEvent.
//
//  Root cause (NVIDIA H100): Leo traces a 3-file dependency chain responsible
//  for 19.8% of stall cycles:
//    CollisionEvent.cc → macroscopicCrossSection() → MacroscopicCrossSection.cc → NuclearData.hh
//  The cross-file call prevents the compiler from reordering loads.
//
//  Root cause (AMD MI300A): Leo identifies flat_load latency through the same
//  cross-file pointer chain, compounded by extreme register pressure
//  (153 VGPRs, 0% occupancy).
//
//  Optimization:
//    1. Inline the macroscopicCrossSection() call — eliminates cross-file barrier
//    2. Add __restrict__ to reaction pointer — enables load reordering
//    3. Hoist cellNumberDensity before isotope loop — avoids redundant loads
//----------------------------------------------------------------------------------------------------------------------

HOST_DEVICE
void updateTrajectory( double energy, double angle, MC_Particle& particle )
{
    particle.kinetic_energy = energy;
    double cosTheta = angle;
    double randomNumber = rngSample(&particle.random_number_seed);
    double phi = 2 * 3.14159265 * randomNumber;
    double sinPhi = sin(phi);
    double cosPhi = cos(phi);
    double sinTheta = sqrt((1.0 - (cosTheta*cosTheta)));
    particle.direction_cosine.Rotate3DVector(sinTheta, cosTheta, sinPhi, cosPhi);
    double speed = (PhysicalConstants::_speedOfLight *
            sqrt((1.0 - ((PhysicalConstants::_neutronRestMassEnergy *
            PhysicalConstants::_neutronRestMassEnergy) /
            ((energy + PhysicalConstants::_neutronRestMassEnergy) *
            (energy + PhysicalConstants::_neutronRestMassEnergy))))));
    particle.velocity.x = speed * particle.direction_cosine.alpha;
    particle.velocity.y = speed * particle.direction_cosine.beta;
    particle.velocity.z = speed * particle.direction_cosine.gamma;
    randomNumber = rngSample(&particle.random_number_seed);
    particle.num_mean_free_paths = -1.0*log(randomNumber);
}
HOST_DEVICE_END

HOST_DEVICE

bool CollisionEvent(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int tally_index)
{
   const MC_Cell_State &cell = monteCarlo->domain[mc_particle.domain].cell_state[mc_particle.cell];

   int globalMatIndex = cell._material;

   //------------------------------------------------------------------------------------------------------------------
   //    Pick the isotope and reaction.
   //------------------------------------------------------------------------------------------------------------------
   double randomNumber = rngSample(&mc_particle.random_number_seed);
   double totalCrossSection = mc_particle.totalCrossSection;
   double currentCrossSection = totalCrossSection * randomNumber;
   int selectedIso = -1;
   int selectedUniqueNumber = -1;
   int selectedReact = -1;
   int numIsos = (int)monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso.size();

   // OPT: Hoist cellNumberDensity out of the isotope loop.
   // Was reloaded inside macroscopicCrossSection() on every call via
   // monteCarlo->domain[domainIndex].cell_state[cellIndex]._cellNumberDensity
   double cellNumberDensity = cell._cellNumberDensity;

   for (int isoIndex = 0; isoIndex < numIsos && currentCrossSection >= 0; isoIndex++)
   {
      int uniqueNumber = monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso[isoIndex]._gid;
      double atomFraction = monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso[isoIndex]._atomFraction;
      int numReacts = monteCarlo->_nuclearData->getNumberReactions(uniqueNumber);

      // OPT: Precompute prefactor (was atomFraction * cellNumberDensity
      // recomputed inside macroscopicCrossSection on every reaction)
      double prefactor = atomFraction * cellNumberDensity;
      if (prefactor == 0.0) continue;

      // OPT: Add __restrict__ to enable compiler load reordering.
      // The cross-file call boundary (CollisionEvent.cc → MacroscopicCrossSection.cc
      // → NuclearData.hh) prevented the compiler from proving non-aliasing.
      NuclearDataReaction * __restrict__ reactions =
          &monteCarlo->_nuclearData->_isotopes[uniqueNumber]._species[0]._reactions[0];

      // OPT: Inlined loop body (was macroscopicCrossSection() call).
      // Eliminates the 3-file cross-file dependency chain that Leo
      // identified as 19.8% of stall cycles on NVIDIA H100.
      for (int reactIndex = 0; reactIndex < numReacts; reactIndex++)
      {
         double xs = reactions[reactIndex].getCrossSection(mc_particle.energy_group);
         currentCrossSection -= prefactor * xs;
         if (currentCrossSection < 0)
         {
            selectedIso = isoIndex;
            selectedUniqueNumber = uniqueNumber;
            selectedReact = reactIndex;
            break;
         }
      }
   }
   qs_assert(selectedIso != -1);

   //------------------------------------------------------------------------------------------------------------------
   //    Do the collision.
   //------------------------------------------------------------------------------------------------------------------
   double energyOut[MAX_PRODUCTION_SIZE];
   double angleOut[MAX_PRODUCTION_SIZE];
   int nOut = 0;
   double mat_mass = monteCarlo->_materialDatabase->_mat[globalMatIndex]._mass;

   monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0]._reactions[selectedReact].sampleCollision(
      mc_particle.kinetic_energy, mat_mass, &energyOut[0], &angleOut[0], nOut, &(mc_particle.random_number_seed), MAX_PRODUCTION_SIZE );

   //--------------------------------------------------------------------------------------------------------------
   //  Post-Collision Phase 1:
   //    Tally the collision
   //--------------------------------------------------------------------------------------------------------------
   QS::atomicIncrement( monteCarlo->_tallies->_balanceTask[tally_index]._collision );
   NuclearDataReaction::Enum reactionType = monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0].\
           _reactions[selectedReact]._reactionType;
   switch (reactionType)
   {
      case NuclearDataReaction::Scatter:
         QS::atomicIncrement( monteCarlo->_tallies->_balanceTask[tally_index]._scatter);
         break;
      case NuclearDataReaction::Absorption:
         QS::atomicIncrement( monteCarlo->_tallies->_balanceTask[tally_index]._absorb);
         break;
      case NuclearDataReaction::Fission:
         QS::atomicIncrement( monteCarlo->_tallies->_balanceTask[tally_index]._fission);
         QS::atomicAdd( monteCarlo->_tallies->_balanceTask[tally_index]._produce, (uint64_t) nOut);
         break;
      case NuclearDataReaction::Undefined:
         printf("reactionType invalid\n");
         qs_assert(false);
   }

   if( nOut == 0 ) return false;

   for (int secondaryIndex = 1; secondaryIndex < nOut; secondaryIndex++)
   {
        MC_Particle secondaryParticle = mc_particle;
        secondaryParticle.random_number_seed = rngSpawn_Random_Number_Seed(&mc_particle.random_number_seed);
        secondaryParticle.identifier = secondaryParticle.random_number_seed;
        updateTrajectory( energyOut[secondaryIndex], angleOut[secondaryIndex], secondaryParticle );
        monteCarlo->_particleVaultContainer->addExtraParticle(secondaryParticle);
   }

   updateTrajectory( energyOut[0], angleOut[0], mc_particle);

   if ( nOut > 1 )
       monteCarlo->_particleVaultContainer->addExtraParticle(mc_particle);

   mc_particle.energy_group = monteCarlo->_nuclearData->getEnergyGroup(mc_particle.kinetic_energy);

   return nOut == 1;
}

HOST_DEVICE_END
