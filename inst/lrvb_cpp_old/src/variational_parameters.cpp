# include "variational_parameters.h"

// The index in a vector of lower diagonal terms of a particular matrix value.
int get_ud_index(int i, int j) {
  // If the column is less than the row it's already an upper diagonal index.
  return j <= i ? (j + i * (i + 1) / 2):
                  (i + j * (j + 1) / 2);
};
