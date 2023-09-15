#!/bin/bash

scancel  $(squeue -u $USER | tail -1| awk '{print $1}')