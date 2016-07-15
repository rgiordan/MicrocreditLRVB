# include "kahan_summation.h"

// Due to a bug in Stan, we can't include their headers here.
template class KahanAccumulator<double>;
// template class var KahanSum;
// template class fvar KahanSum;
