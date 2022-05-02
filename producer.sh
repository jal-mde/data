#!/usr/bin/bash
declare i
declare x
for i in {0..80}
        do
                x=$((($i*10)+1))
                sed -n ''$x',+9p' pokemon.csv
                sleep 2


        done
