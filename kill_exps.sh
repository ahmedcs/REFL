#!/usr/bin/env bash

if [[ ! -z $1  &&  ! -z $2 ]];
then
  python core/evals/manager.py stop $1 $2
else
  if [[ ! -z $1 ]];
  then
    python core/evals/manager.py stop $1
  else
     python core/evals/manager.py stop 'all'
  fi
fi