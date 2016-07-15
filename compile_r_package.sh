#!/bin/bash

PACKAGE_LOC=$GIT_REPO_LOC"/MicrocreditLRVB"
echo 'library(devtools); library(Rcpp); compileAttributes("'$PACKAGE_LOC'"); install_local("'$PACKAGE_LOC'")' | R --vanilla
