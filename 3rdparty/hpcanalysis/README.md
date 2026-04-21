<!--
SPDX-FileCopyrightText: Contributors to the HPCToolkit Project

SPDX-License-Identifier: CC-BY-4.0
-->

> **Vendored for SC26 AE.** This is a snapshot of `hpcanalysis` from
> `gitlab.com/yuningxia/hpcanalysis` (branch `feature/gins-stall-support`)
> at commit `39473757b82358640465a11c68383c7ba49b9f13` (2026-02-05).
> The `build/` directory is excluded; it will be regenerated on `uv sync`.

# hpcanalysis

Framework for analyzing performance of applications at exascale

## Overview

The `hpcanalysis` framework provides a scalable and efficient solution for processing and analyzing large-scale out-of-core performance datasets collected by HPCToolkit. It enables users to programmatically inspect performance data, automate analysis tasks, and gain insights into application behavior on exascale supercomputers.

## Features

* **Programmatic Access:** Enables users to write Python programs to analyze HPCToolkit data.
* **Pandas Integration:** Leverages the Pandas library for data manipulation and analysis.
* **Jupyter Notebooks:** Supports automation, integration, and saving of analysis tasks within Jupyter notebooks.
* **Scalable Data Handling:** Implements techniques for pruning code regions with low information content and sampling large-scale executions using specialized query expressions.
* **Memory Efficiency:** Designed to be memory-efficient by fetching and storing data only when requested.

## HPCToolkit

- `HPCToolkit` is a measurement tool for profiling and tracing HPC applications on heterogeneous exascale supercomputers.   
- `hpcanalysis` is a framework that enables users to analyze these measurements, which can grow to terabytes, by providing techniques for automated and programmatic data inspection.   

## Architecture

The `hpcanalysis` framework is structured hierarchically, and consists of several layers:

* **Read API and Query API:** Interfaces for extracting data.
* **Data Analysis:** Functions for performing common analysis tasks.

## Data Extraction

### Small Metadata

* **Metric Descriptions:** Entire data section extracted and stored in a Pandas DataFrame table.
* **Profile Descriptions:** Entire data section extracted and stored in a Pandas DataFrame table.

### Large Metadata

* **Calling Context Tree (CCT):** Extracted and stored in a Pandas DataFrame table, with support for pruning code regions with low information content.

### Large Performance Data

* **Performance Profiles:** Extracted and stored in a Pandas DataFrame table, with support for sampling profiles by execution context, calling contexts, and metrics.
* **Execution Traces:** Extracted and stored in a Pandas DataFrame table, with support for sampling traces by execution context and time intervals.

## Installation

```
python3 -m pip install git+https://gitlab.com/hpctoolkit/hpcanalysis.git
```

## Examples

### Extract Metadata

```python
>>> import hpcanalysis
>>> query_api = hpcanalysis.open_db("path/to/database")._query_api
>>> query_api.query_metric_descriptions("cputime (i,e)")        # extract CPU time metric with inclusive and exclusive scope
        id      name            aggregation     scope   unit
1       1       cputime         prop            e       sec
5       1       cputime         sum             e       sec
3       3       cputime         prop            i       sec
7       3       cputime         sum             i       sec
>>> query_api.query_profile_descriptions("rank(0).thread(1-8)") # extract profiles with rank ID 0 and thread IDs 1,2
        node            rank    thread  core    gpustream       gpucontext      gpudevice       cct_samples     metric_values   trace_samples
id
1       2148202605      0       1       <NA>        <NA>            <NA>         <NA>           100             204             20001
2       2148202609      0       2       <NA>        <NA>            <NA>         <NA>           132             290             20036
3       2148202602      0       3       <NA>        <NA>            <NA>         <NA>           57              124             19863
4       2148202605      0       4       <NA>        <NA>            <NA>         <NA>           88              176             19826
5       2148202615      0       5       <NA>        <NA>            <NA>         <NA>           145             333             20035
6       2148202605      0       6       <NA>         1               5           <NA>           0               0               2422
7       2148202602      0       7       <NA>        <NA>            <NA>         <NA>           99              192             19972
8       2148202613      0       8       <NA>        <NA>            <NA>         <NA>           143             325             20038
>>> query_api.query_cct("function(foo).line(foo.c:12)")         # extract specific call paths from the calling context tree
        type            parent  children        depth   name    file_path       line    module_path     offset
id
996     function        994     [1006]          1       42800   41944           0       41640           56957072
1006    line            996     []              2       <NA>    41944           0       <NA>            <NA>
1038    function        994     [1073]          1       62960   41944           0       41640           57000256
1073    line            1038    []              2       <NA>    41944           0       <NA>            <NA>
660     function        994     []              1       <NA>    <NA>            <NA>    <NA>            <NA>
1130    function        994     [1162]          1       61640   41944           0       41640           57040592
1162    line            1130    []              2       <NA>    <NA>            0       <NA>            <NA>
274     instruction     994     []              1       <NA>    <NA>            <NA>    41640           56911168
```

### Extract Slices of Performance Data

```python
>>> import hpcanalysis
>>> query_api = hpcanalysis.open_db("path/to/database")._query_api
>>> query_api.query_profile_slices(
    "rank(0).thread(1,5-7)",    # extract profiles with rank ID 0 and thread IDs 1,5,6,7    
    "function(MPI_*)",          # extract MPI functions
    "cputime (i)",              # extract CPU time metric with inclusive scope
)
                                value
profile_id  cct_id  metric_id
3           3       3           0.010015        
                    3           0.010015
                    3           0.035079
            4       3           0.025064     
                    3           0.025064        
                    3           0.025064
            5       3           0.020057
                    3           0.020057
                    3           0.065163
            7       3           99.925522
>>> query_api.query_trace_slices(
    "rank(0).thread(1,5-7)",    # extract traces with rank ID 0 and thread IDs 1,5,6,7    
    (1713838621, 1713910621),   # extract specific time interval
)
        profile_id      cct_id  start_timestamp         end_timestamp
225     56              0       1685978417124857000     1685978417151116000        
229     5               0       1685978417124919000     1685978417151155000         
231     8               0       1685978417124936000     1685978417151135000    
227     29              0       1685978417124971000     1685978417151172000 
293     13              0       1685978417124972000     1685978417151161000 
289     40              0       1685978417124978000     1685978417155013000
291     64              0       1685978417124988000     1685978417151078000    
295     43              0       1685978417124993000     1685978417151140000 
```

### Reduce Calling Context Tree

```python
>>> import hpcanalysis
>>> cct_reduction = CCTReduction()
>>> cct_reduction.add_reduction(TimeReduction(percentage_threshold=1))  # remove nodes with little inclusive cost
>>> cct_reduction.add_reduction(MPIReduction())                         # remove implementation details of MPI
>>> cct_reduction.add_reduction(FunctionReduction())                    # remove nodes that are not functions
>>> hpc_api = hpcanalysis.open_db("path/to/database", cct_reduction=cct_reduction)
```

### Detect Load Balance

```python
>>> import hpcanalysis
>>> hpc_api = hpcanalysis.open_db("path/to/database")
>>> hpc_api.load_balance("function(MPI_*)", "rank(0-65536:4096)")       # extract every 4096th rank
                load balance
function
MPI_Scan        1.000000
MPI_Reduce      0.600000
MPI_Bcast       0.336671
MPI_Barrier     0.286229
MPI_Sendrecv    0.215673
MPI_Irecv       0.192319
MPI_Send        0.155680
>>> hpc_api.load_balance("function(MPI_*)", "rank(0-65536:512)")        # extract every 512th rank
                load balance
function
MPI_Scan        0.997816
MPI_Reduce      0.434601
MPI_Bcast       0.234888
MPI_Barrier     0.280955
MPI_Sendrecv    0.209480
MPI_Irecv       0.153511
MPI_Send        0.143607
>>> hpc_api.load_balance("function(MPI_*)", "rank(8-65536:64)")         # extract every 64th rank
                load balance
function
MPI_Scan        0.976709
MPI_Reduce      0.327811
MPI_Bcast       0.216440
MPI_Barrier     0.277750
MPI_Sendrecv    0.203069
MPI_Irecv       0.144470
MPI_Send        0.138225
```

### Validate Metadata

```python
import hpcanalysis

hpc_api = hpcanalysis.open_db (DB_PATH)
profiles = hpc_api._query_api.query_profile_descriptions("rank")

assert len(profiles) == 65536                   # there are in total 65,536 execution profiles
assert len(profiles ["node"].unique()) == 8192  # execution is performed on 8,192 compute nodes

for node in profiles ["node"].unique().tolist():
    assert len(profiles[profiles["node"] == node]["rank"].unique())  == 8       # each node has 8 ranks executing on it
    assert len(profiles[profiles["node"] == node]["core"].unique()) == 8        # each rank is executed on a separate core
    assert len(profiles[profiles["node"] == node]["thread"].unique())  == 1     # each rank operates with a single thread
```

### HPCReport

```python
>>> import hpcanalysis
>>> hpc_api = hpcanalysis.open_db("path/to/database")
>>> hpc_api.hpcreport()
                                                                        TIME            PERCENTAGE (%)
MAJOR           MINOR		
CPU (sec)	CPU total	                                        4008.423769	100.00
                OpenMP Idle	                                        2399.906262	59.87
                MPI Point-to-Point	                                803.498280	20.05
                MPI Collective	                                        174.333809	4.35
                MPI Environmental Inquiry Functions and Profiling       0.031208        0.00
GPU (sec)	GPU total	                                        608.093728	100.00
                GPU kernel execution	                                608.001020	99.98
                GPU explicit data copy	                                0.092707	0.02
hpc_api.hpcreport(verbose=True)
		                                                        TIME	        PERCENTAGE (%)
MAJOR	        MINOR		
CPU (sec)	CPU total	                                        4008.423769	100.00
                <omp idle>	                                        2399.906262	59.87
                MPI_Recv	                                        803.218108	20.04
                MPI_Allreduce	                                        165.956858	4.14
                MPI_Bcast	                                        8.371937	0.21
                MPI_Ssend	                                        0.210093	0.01
                MPI_Send	                                        0.070079	0.00
                MPI_Finalize	                                        0.031208	0.00
                MPI_Barrier	                                        0.005014	0.00
GPU (sec)	GPU total	                                        608.093728	100.00
                GPU kernel execution	                                608.001020	99.98
                GPU explicit data copy	                                0.092707	0.02
```

### Detect GPU Idleness

```python
>>> import hpcanalysis
>>> hpc_api = hpcanalysis.open_db("path/to/database")
>>> hpc_api.gpu_idleness()
        GPU total time  GPU idle time
rank		
1	96.033223	4.239003
0	89.908101	10.369471
2	77.191926	23.074498
3	75.432494	24.829490
4	68.689743	31.583649
5	68.578856	31.698608
7	67.068608	33.208211
6	65.190776	35.076447
```

### Create Flat Profile

```python
>>> import hpcanalysis
>>> hpc_api = hpcanalysis.open_db("path/to/database")
>>> hpc_api.flat_profile("function(<omp idle>)", "rank")
        <omp idle> (sec)        
rank		
0	1101.564123	        
1	1101.682350	        
2	1101.471012	        
3	1101.967589	        
4	1102.064512	        
5	1102.000164	        
6	1101.420559	        
7	1101.799385
>>> hpc_api.flat_profile("function(<omp idle>)", "rank", include_percentage=True)
        <omp idle> (sec)        percentage (%)
rank		
0	1101.564123	        74.54
1	1101.682350	        74.55
2	1101.471012	        74.54
3	1101.967589	        74.56
4	1102.064512	        74.56
5	1102.000164	        74.57
6	1101.420559	        74.53
7	1101.799385	        74.55
>>> hpc_api.flat_profile("function(<omp idle>)", "rank(0,1).thread", include_percentage=True)
                <omp idle> (sec)        percentage (%)
rank    thread		
0	1	183.652248	        99.32
        2	183.666212	        99.33
        3	183.528916	        99.26
        4	183.470563	        99.23
        5	183.565956	        99.28
        6	183.680228	        99.33
1	1	183.684245	        99.34
        2	183.634589	        99.32
        3	183.544208	        99.26
        4	183.607711	        99.30
        5	183.639453	        99.31
        6	183.572144	        99.27
>>> hpc_api.flat_profile("function(MPI_*)", include_percentage=True)
                        time (sec)      percentage (%)
rank    function		
0	MPI_Allreduce	6.819039	1.70
        MPI_Barrier	0.005014	0.00
        MPI_Bcast	0.275615	0.07
        MPI_Recv	0.010004	0.00
        MPI_Ssend	0.026194	0.01
1	MPI_Allreduce	0.881706	0.22
        MPI_Bcast	1.122408	0.28
        MPI_Finalize	0.010458	0.00
        MPI_Send	0.010020	0.00
        MPI_Ssend	0.026207	0.01
2	MPI_Allreduce	19.269688	4.81
        MPI_Bcast	1.167502	0.29
        MPI_Send	0.010009	0.00
        MPI_Ssend	0.026212	0.01
```

### Combine Data

```python
>>> import hpcanalysis
>>> hpc_api = hpcanalysis.open_db("path/to/database")
>>> hpc_api.hpc_api.gpu_idleness().merge(
        hpc_api.flat_profile("function(MPI_Barrier)"), 
                how="left", left_index=True, right_index=True)
                .sort_values("GPU idle time")
        GPU total time  GPU idle time   MPI_Barrier (sec)       percentage (%)
rank				
0	87.782469	96.089508	4.708358	        0.32
2	87.380800	96.410952	5.107981	        0.35
3	86.981624	97.044456	5.057824	        0.34
5	86.700436	97.145361	4.817616	        0.33
1	86.171396	97.599539	5.112805	        0.35
4	85.713555	98.260222	5.092756	        0.34
7	85.543837	98.359067	5.073124	        0.34
6	85.175730	98.799588	4.807655	        0.33
```

## Local Development Setup

Make sure you have [PDM](https://pdm-project.org/) installed. If you haven't already, you can install it using pip:

```bash
pip install pdm
```

Then, install the project's development dependencies:

```bash
pdm install --group dev
```

### Formatting, Linting, and Testing

We use PDM scripts to manage code formatting, linting, and testing. You can easily run these checks locally before submitting a pull request.

- `pdm run format`: Automatically formats the code using Black.
- `pdm run lint`: Checks the code for style issues and potential errors using Flake8 and MyPy.
- `pdm run test`: Runs the unit tests using Pytest and checks code coverage with Pytest-cov.
- `pdm run all`: Executes all the above checks (format, lint, and test).

