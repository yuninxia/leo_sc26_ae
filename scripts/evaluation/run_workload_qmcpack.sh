#!/bin/bash
# Run QMCPACK workload test on all three GPU vendors
#
# Usage:
#   ./run_workload_qmcpack.sh              # Run on all vendors
#   ./run_workload_qmcpack.sh nvidia       # Run on a single vendor
#   ./run_workload_qmcpack.sh amd intel    # Run on specific vendors
#
# Runs the H2O molecule example (8 electrons, pseudopotentials, VMC + DMC)
# using the batched driver inside the leo-qmcpack-{vendor} Docker container.
# This is a realistic workload that exercises the full QMCPACK pipeline.
#
# For heavier benchmarks (NiO solid-state), see:
#   /opt/qmcpack-src/tests/performance/NiO/README  (requires HDF5 wavefunction data)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# Vendor configuration
# ============================================
declare -A MACHINES DOCKER_GPU_FLAGS OMP_THREADS
MACHINES[nvidia]="illyad"
MACHINES[nvidia-arm]="hopper1"
MACHINES[amd]="odyssey"
MACHINES[intel]="headroom"

DOCKER_GPU_FLAGS[nvidia]="--gpus all"
DOCKER_GPU_FLAGS[nvidia-arm]="--gpus all"
DOCKER_GPU_FLAGS[amd]="--device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video"
DOCKER_GPU_FLAGS[intel]="--device=/dev/dri"

# QMCPACK creates one crowd per OMP thread, and each crowd allocates a
# cuSolver/hipBLAS handle (~450 MB GPU memory each). Too many handles exhaust
# GPU resources (GitHub issue #2874). We cap OMP_NUM_THREADS to 4 on all
# vendors. The crowd count is also explicitly set in the XML input below.
OMP_THREADS[nvidia]="OMP_NUM_THREADS=4"
OMP_THREADS[nvidia-arm]="OMP_NUM_THREADS=4"
OMP_THREADS[amd]="OMP_NUM_THREADS=4"
OMP_THREADS[intel]="OMP_NUM_THREADS=4"

# ============================================
# QMCPACK H2O workload (batched driver)
# ============================================
# The H2O example ships with QMCPACK and includes:
#   - 8 electrons (4 up, 4 down)
#   - Pseudopotentials (BFD for O and H)
#   - HF wavefunction with Jastrow factors
#   - VMC sampling followed by DMC propagation
#
# We convert from the legacy driver to the batched driver for GPU execution.
# The input and supporting files are at /opt/qmcpack-src/examples/molecules/H2O/

# This heredoc creates a batched-driver version of the H2O example.
# It references H2O.HF.wfs.xml, O.BFD.xml, H.BFD.xml from the same directory.
QMCPACK_INPUT='<?xml version="1.0"?>
<simulation>
  <project id="H2O" series="1">
    <application name="qmcapp" role="molecu" class="serial" version="0.2">
      H2O molecule - batched driver for GPU verification
    </application>
  </project>

  <random seed="42"/>

  <particleset name="e">
    <group name="u" size="4">
      <parameter name="charge">-1</parameter>
      <attrib name="position" datatype="posArray">
        2.9151687332e-01 -6.5123272502e-01 -1.2188463918e-01
        5.8423636048e-01  4.2730406357e-01 -4.5964306231e-03
        3.5228575807e-01 -3.5027014639e-01  5.2644808295e-01
       -5.1686250912e-01 -1.6648002292e+00  6.5837023441e-01
      </attrib>
    </group>
    <group name="d" size="4">
      <parameter name="charge">-1</parameter>
      <attrib name="position" datatype="posArray">
        3.1443445436e-01  6.5068682609e-01 -4.0983449009e-02
       -3.8686061749e-01 -9.3744432997e-02 -6.0456005388e-01
        2.4978241724e-02 -3.2862514649e-02 -7.2266047173e-01
       -4.0352404772e-01  1.1927734805e+00  5.5610824921e-01
      </attrib>
    </group>
  </particleset>

  <particleset name="ion0" size="3">
    <group name="O">
      <parameter name="charge">6</parameter>
      <parameter name="valence">4</parameter>
      <parameter name="atomicnumber">8</parameter>
    </group>
    <group name="H">
      <parameter name="charge">1</parameter>
      <parameter name="valence">1</parameter>
      <parameter name="atomicnumber">1</parameter>
    </group>
    <attrib name="position" datatype="posArray">
      0.0000000000e+00  0.0000000000e+00  0.0000000000e+00
      0.0000000000e+00 -1.4308249289e+00  1.1078707576e+00
      0.0000000000e+00  1.4308249289e+00  1.1078707576e+00
    </attrib>
    <attrib name="ionid" datatype="stringArray">
      O H H
    </attrib>
  </particleset>

  <include href="H2O.HF.wfs.xml"/>

  <hamiltonian name="h0" type="generic" target="e">
    <pairpot name="ElecElec" type="coulomb" source="e" target="e"/>
    <pairpot name="ELEMENT-ECP" type="pseudo" source="ion0" target="e" format="xml" wavefunction="psi0">
      <pseudo elementType="O" format="xml" href="O.BFD.xml"/>
      <pseudo elementType="H" format="xml" href="H.BFD.xml"/>
    </pairpot>
    <constant name="IonIon" type="coulomb" source="ion0" target="ion0"/>
  </hamiltonian>

  <init source="ion0" target="e"/>

  <!-- VMC sampling with batched driver -->
  <!-- crowds=4 to limit cuSolver handle count (each ~450 MB GPU RAM, GH #2874) -->
  <qmc method="vmc_batch" move="pbyp">
    <estimator name="LocalEnergy" hdf5="no"/>
    <parameter name="crowds">4</parameter>
    <parameter name="total_walkers">16</parameter>
    <parameter name="warmupSteps">50</parameter>
    <parameter name="substeps">5</parameter>
    <parameter name="steps">10</parameter>
    <parameter name="blocks">5</parameter>
    <parameter name="timestep">0.3</parameter>
    <parameter name="usedrift">no</parameter>
  </qmc>

  <!-- DMC propagation with batched driver -->
  <qmc method="dmc_batch" move="pbyp" checkpoint="-1">
    <estimator name="LocalEnergy" hdf5="no"/>
    <parameter name="crowds">4</parameter>
    <parameter name="total_walkers">64</parameter>
    <parameter name="warmupSteps">20</parameter>
    <parameter name="timestep">0.005</parameter>
    <parameter name="steps">5</parameter>
    <parameter name="blocks">10</parameter>
    <parameter name="nonlocalmoves">yes</parameter>
  </qmc>
</simulation>'

# ============================================
# Determine vendors to run
# ============================================
if [ $# -eq 0 ]; then
    VENDORS=(nvidia amd intel)
else
    VENDORS=("$@")
fi

# ============================================
# Run a single vendor (called as a function)
# ============================================
# Writes result to a temp file for the parent to collect
run_vendor() {
    local VENDOR="$1"
    local OUTFILE="$2"

    local MACHINE="${MACHINES[$VENDOR]}"
    local GPU_FLAGS="${DOCKER_GPU_FLAGS[$VENDOR]}"
    local OMP_ENV="${OMP_THREADS[$VENDOR]}"
    local IMAGE="leo-qmcpack-${VENDOR}:latest"

    echo "[$VENDOR] Starting on $MACHINE ($IMAGE)..."

    local OUTPUT
    if OUTPUT=$(ssh "$MACHINE" "docker run --rm \
        $GPU_FLAGS \
        --entrypoint='' \
        $IMAGE \
        bash -c 'cd /opt/qmcpack-src/examples/molecules/H2O && cat > h2o_batch.xml << '\"'\"'XMLEOF'\"'\"'
${QMCPACK_INPUT}
XMLEOF
${OMP_ENV} qmcpack h2o_batch.xml 2>&1'" 2>&1); then

        if echo "$OUTPUT" | grep -q "QMCPACK execution completed successfully"; then
            local ENERGY=$(echo "$OUTPUT" | grep "reference energy" | tail -1 | awk '{print $NF}')
            local VARIANCE=$(echo "$OUTPUT" | grep "reference variance" | tail -1 | awk '{print $NF}')
            local EXEC_TIME=$(echo "$OUTPUT" | grep "Total Execution time" | sed 's/.*= *//' | tr -d ' ')
            echo "PASS" > "$OUTFILE"
            echo "$ENERGY" >> "$OUTFILE"
            echo "$VARIANCE" >> "$OUTFILE"
            echo "$EXEC_TIME" >> "$OUTFILE"
        else
            echo "FAIL" > "$OUTFILE"
            echo "$OUTPUT" | tail -20 >> "$OUTFILE"
        fi
    else
        echo "FAIL" > "$OUTFILE"
        echo "$OUTPUT" | tail -20 >> "$OUTFILE"
    fi
}

# ============================================
# Launch all vendors in parallel
# ============================================
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
PIDS=()

echo "Launching QMCPACK H2O workload on ${#VENDORS[@]} vendor(s) in parallel..."
echo ""

for VENDOR in "${VENDORS[@]}"; do
    if [[ -z "${MACHINES[$VENDOR]+_}" ]]; then
        echo "ERROR: Unknown vendor '$VENDOR' (valid: nvidia, amd, intel)"
        exit 1
    fi
    run_vendor "$VENDOR" "$TMPDIR/$VENDOR.result" &
    PIDS+=($!)
done

# Wait for all to finish
for PID in "${PIDS[@]}"; do
    wait "$PID" 2>/dev/null || true
done

# ============================================
# Collect and display results
# ============================================
PASS=0
FAIL=0
RESULTS=()

for VENDOR in "${VENDORS[@]}"; do
    MACHINE="${MACHINES[$VENDOR]}"
    IMAGE="leo-qmcpack-${VENDOR}:latest"
    RESULT_FILE="$TMPDIR/$VENDOR.result"

    echo "============================================"
    echo " QMCPACK H2O workload: $VENDOR on $MACHINE"
    echo " Image: $IMAGE"
    echo "============================================"

    if [ ! -f "$RESULT_FILE" ]; then
        echo "  Status:   FAIL (no result file)"
        RESULTS+=("$VENDOR: FAIL")
        FAIL=$((FAIL + 1))
    elif [ "$(head -1 "$RESULT_FILE")" = "PASS" ]; then
        ENERGY=$(sed -n '2p' "$RESULT_FILE")
        VARIANCE=$(sed -n '3p' "$RESULT_FILE")
        EXEC_TIME=$(sed -n '4p' "$RESULT_FILE")
        echo "  Status:   PASS"
        echo "  Energy:   $ENERGY Ha"
        echo "  Variance: $VARIANCE"
        echo "  Time:     $EXEC_TIME"
        RESULTS+=("$VENDOR: PASS (E=$ENERGY Ha, t=$EXEC_TIME)")
        PASS=$((PASS + 1))
    else
        echo "  Status:   FAIL"
        echo "  Output (last 20 lines):"
        tail -n +2 "$RESULT_FILE" | sed 's/^/    /'
        RESULTS+=("$VENDOR: FAIL")
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

# ============================================
# Summary
# ============================================
echo "============================================"
echo " Summary: $PASS passed, $FAIL failed"
echo "============================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

if [ $FAIL -gt 0 ]; then
    exit 1
fi
