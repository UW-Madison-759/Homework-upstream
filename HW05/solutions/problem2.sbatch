#!/bin/sh

#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --error=/srv/home/thaines/HW05/problem2.err
#SBATCH --output=/srv/home/thaines/HW05/problem2.out

for f in problem2 problem2_unrolled
do
	rm -f $f.dat
	for i in $(seq 1 40)
	do
		./$f $i >> $f.dat
	done
done