#!/bin/sh

EXE_PATH=./x64/Release/OptymalizacjaCUDA.exe

for i in 1 2 4 8 16 32 64 128 256 512 1024; do
   for j in 1 2 4 8 16 32 64 128 256 512 1024; do
      if [ $((i * j)) -le 1024 ]; then
         printf "%i %i " $i $j
         ${EXE_PATH} 100 10000 1 $i $j
      fi
   done
done


#100 20000 7 128 1 0.2