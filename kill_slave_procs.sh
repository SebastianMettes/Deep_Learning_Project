#/bin/bash
ps -ef |grep slave_node.py | awk '{print $2}' | xargs -I {} kill {}
