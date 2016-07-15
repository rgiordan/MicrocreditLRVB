# ifndef TEST_EIGEN_H
# define TEST_EIGEN_H

// Tests for Eigen types.
#define EXPECT_VECTOR_EQ(x, y) \
  ASSERT_EQ(x.size(), y.size()); \
  for (int i = 0; i < x.size(); i ++) { \
    EXPECT_DOUBLE_EQ(x(i), y(i)) << " at index " << i; \
  }


#define EXPECT_MATRIX_EQ(x, y) \
  ASSERT_EQ(x.rows(), y.rows()); \
  ASSERT_EQ(x.cols(), y.cols()); \
  for (int i = 0; i < x.rows(); i ++) { \
    for (int j = 0; j < x.rows(); j ++) { \
      EXPECT_DOUBLE_EQ(x(i, j), y(i, j)) << " at index " << i << ", " << j; \
    } \
  }


// Tests for Eigen types.
#define EXPECT_FLOAT_VECTOR_EQ(x, y) \
  ASSERT_EQ(x.size(), y.size()); \
  for (int i = 0; i < x.size(); i ++) { \
    EXPECT_FLOAT_EQ(x(i), y(i)) << " at index " << i; \
  }


#define EXPECT_FLOAT_MATRIX_EQ(x, y) \
  ASSERT_EQ(x.rows(), y.rows()); \
  ASSERT_EQ(x.cols(), y.cols()); \
  for (int i = 0; i < x.rows(); i ++) { \
    for (int j = 0; j < x.rows(); j ++) { \
      EXPECT_FLOAT_EQ(x(i, j), y(i, j)) << " at index " << i << ", " << j; \
    } \
  }

# endif
