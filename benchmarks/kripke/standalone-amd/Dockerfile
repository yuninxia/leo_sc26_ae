# Standalone Kripke build for AMD MI300A (gfx942)
# No dependencies on leo-* images — builds directly from public ROCm image.
#
# Build:  docker build -t kripke-amd .
# Run:    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video kripke-amd
#
FROM rocm/dev-ubuntu-22.04:6.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git make g++ && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --recursive https://github.com/LLNL/Kripke.git /opt/kripke-src

RUN mkdir -p /opt/kripke-build && cd /opt/kripke-build && \
    cmake /opt/kripke-src \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake" \
        -DENABLE_HIP=On \
        -DENABLE_CHAI=On \
        -DCHAI_DISABLE_RM=OFF \
        -DCHAI_THIN_GPU_ALLOCATE=OFF \
        -DCHAI_ENABLE_UM=OFF \
        -DENABLE_MPI=Off \
        -DENABLE_OPENMP=Off \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DROCM_PATH=/opt/rocm \
    && make -j$(nproc) \
    && mkdir -p /opt/kripke/bin \
    && cp kripke.exe /opt/kripke/bin/ 2>/dev/null \
    || find /opt/kripke-build -name kripke.exe -exec cp {} /opt/kripke/bin/ \;

ENV PATH="/opt/kripke/bin:${PATH}"

CMD ["kripke.exe", "--zones", "32,32,32", "--groups", "32", "--niter", "10"]
