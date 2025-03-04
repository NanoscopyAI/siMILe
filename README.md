### Source code for SiMiLe-M paper


- Build singularity container:
  ````
  singularity build singularity/similem.sif singularity/similem.def
  ````

- Test container, should get the help dialog:
  ```
  singularity exec singularity/similem.sif python run.py -h
  ```
