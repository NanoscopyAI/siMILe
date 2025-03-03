
Build singularity container:     

`singularity build similem.sif similem.def`

Test container, should get the help dialog:      
`singularity exec similem>.sif python CHWOS/run.py`
