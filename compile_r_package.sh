#!/bin/bash

echo "\n\n\n"
echo $(date)
pushd .
PACKAGE_LOC=$GIT_REPO_LOC"/MicrocreditLRVB"
cd $PACKAGE_LOC"/inst/microcredit_cpp/build/"
sudo make install
if [ $? -ne 0 ]; then
   echo "Exiting."
   popd
   exit 1
fi
cd $PACKAGE_LOC
echo 'library(devtools); library(Rcpp); compileAttributes("'$PACKAGE_LOC'"); install_local("'$PACKAGE_LOC'")' | R --vanilla
popd

echo $(date)
echo "\n\n\n"

