#/bin/bash

echo Launching $1 slave nodes
for ((i=1; i<=$1; i++))
do
  echo "Lauching Slave $i"
  python slave_node.py &
done
