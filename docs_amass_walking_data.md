# AMASS Walking Data For HumEnv

Use the official AMASS site after login:

https://amass.is.tue.mpg.de/download.php

Download **SMPL-H / gender based** archives. For this exoskeleton walking task,
start with these packages:

1. `KIT.tar.bz2` - best first choice; descriptive walking clips.
2. `CMU.tar.bz2` - large locomotion coverage.
3. `BMLmovi.tar.bz2` - natural daily motions, includes walking.
4. `BMLrub.tar.bz2` - walking/running style locomotion.
5. `MPI_HDM05.tar.bz2` - clean locomotion motions.
6. `Transitions.tar.bz2` - transitions that include walk segments; useful later.

Place downloaded archives in either:

```bash
/home/yhzhu/AI/humenv/data_preparation/AMASS/datasets/
```

or:

```bash
/home/yhzhu/AI/humenv/data_preparation/AMASS/archives/
```

Then run:

```bash
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py check
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py extract
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py select
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py process --num-workers 0
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py validate
```

For a quick smoke run:

```bash
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py select --max-files 3
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py process --num-workers 0
conda run --no-capture-output -n mjwarp_env python scripts/amass_walking_pipeline.py validate
```

The converted HDF5 files will be written to:

```bash
/home/yhzhu/AI/humenv/data_preparation/humenv_amass_walking_forward/
```

The quality manifest will be:

```bash
/home/yhzhu/AI/humenv/data_preparation/humenv_amass_walking_forward/walking_manifest.csv
```

Use `show_mocap_track.sh` to visually inspect accepted clips:

```bash
MOCAP_MOTION=/path/to/accepted_motion.hdf5 ./show_mocap_track.sh
```
