#!/bin/sh

make clean

make

echo
echo "*************************"
echo "     Cleaned & Made      "
echo "*************************"
echo


echo
echo "*************************"
echo "     Execute Program     "
echo "*************************"
echo

./matrix 2

echo
echo "*************************"
echo "     Execute NCU         "
echo "*************************"
echo


ncu --metrics sm__warps_active.avg.per_cycle_active,sm__warps_active.max.per_cycle_active,sm__maximum_warps_avg_per_active_cycle ./matrix 2
