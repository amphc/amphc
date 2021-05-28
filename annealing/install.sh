#!/bin/sh

pip install typed_ast typed_astunparse islpy sympy numpy ray $*

echo 'Set followings:'
echo 'export PYTHONPATH=$PWD:$PYTHONPATH'
echo 'export PATH=$PWD:$PATH'
