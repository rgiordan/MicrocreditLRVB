#MicrocreditLRVB

**Warning: this package is currently in a state of transition and
may not be working correctly.  I'm not keeping it up-to-date
since I presume not a lot of people are looking at it.
Please email the author if you're interested in running the examples.**

This is an R package, but it has a number of C++ dependencies that
must be installed and built before it can be installed, including one
in the `inst` directory of this repo.

First, it requires an environment variable called GIT_REPO_LOC to be
set to the location of git repositories.  For example, in your .bashrc,
put the line

```bash
export GIT_REPO_LOC="$HOME/git_repos/"
```

Install stan, stan math, and LinearResponseVariationalBayes.cpp to the git
repo location.

* [Stan repo](https://github.com/stan-dev/stan)
* [Stan math repo](https://github.com/stan-dev/math)
* [LinearResponseVariationalBayes.cpp repo](https://github.com/rgiordan/LinearResponseVariationalBayes.cpp)

You will also need the R packages `Rcpp`, `RcppEigen` and `devtools`.  In R:

```R
install.packages("devtools")
install.packages("Rcpp")
install.packages("RcppEigen")
```

Next you must install the C++ libraries in this repository.  Check out this
repository, change to the subdirectory MicrocreditLRVB/inst/microcredit_cpp/src and
run cmake then make install:

```bash
cd $GIT_REPO_LOC/MicrocreditLRVB/inst/microcredit_cpp/src
cmake .
sudo make install
```

Then install the R package:

```bash
cd $GIT_REPO_LOC
R -e 'library(devtools); install_local("MicrocreditLRVB")'
```

To run the example, first run the stan code in
`inst/R/microcredit_stan_analysis.R`.
This is slow, and will save a re-usable datafile.  Then run the LRVB part with
the script
`inst/R/microcredit_data_analysis.R`.
