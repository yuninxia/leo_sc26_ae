#!/bin/bash
# Prepare QMCPACK NiO-S4 benchmark for GPU profiling (~2-3 min)
# 16 atoms, 192 electrons
# Must be run in the working directory

NIO_SRC="/opt/qmcpack-src/tests/performance/NiO"
H5_HOST="/opt/leo-host/benchmarks/qmcpack"

# Link H5 and pseudopotential files
ln -sf "$H5_HOST/NiO-fcc-supertwist111-supershift000-S4.h5" .
cp "$NIO_SRC/Ni.opt.xml" . 2>/dev/null || true
cp "$NIO_SRC/O.xml" . 2>/dev/null || true

# Copy template and substitute variables
cp "$NIO_SRC/sample/dmc-a16-e192-cpu/NiO-fcc-S4-dmc.xml.in" nio_gpu_profile.xml

# Substitute cmake variables
sed -i "s|\${H5_DIR}|.|g" nio_gpu_profile.xml
sed -i "s|\${PP_DIR}|.|g" nio_gpu_profile.xml
sed -i "s|\${SRC_DIR}|.|g" nio_gpu_profile.xml

# Switch to batched driver for GPU
sed -i 's/legacy/batched/' nio_gpu_profile.xml

# Replace walkers with walkers_per_rank
sed -i 's|<parameter name="walkers">                1 </parameter>|<parameter name="walkers_per_rank">    16 </parameter>|g' nio_gpu_profile.xml

# Increase VMC steps for more profiling data
sed -i '0,/<parameter name="steps">                  1 </{s/<parameter name="steps">                  1 </<parameter name="steps">                 50 </}' nio_gpu_profile.xml
sed -i '0,/<parameter name="blocks">                 2 </{s/<parameter name="blocks">                 2 </<parameter name="blocks">                20 </}' nio_gpu_profile.xml

export OMP_NUM_THREADS=4
echo "NiO-S4 prepared: 16 atoms, 192 electrons, batched driver, 16 walkers"
grep -E "driver_version|walkers|blocks|steps" nio_gpu_profile.xml | head -10
