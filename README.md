
Build singularity container:     

`singularity build singularity/similem.sif singularity/similem.def`

Test container, should get the help dialog:      
`singularity exec similem>.sif python CHWOS/run.py`
