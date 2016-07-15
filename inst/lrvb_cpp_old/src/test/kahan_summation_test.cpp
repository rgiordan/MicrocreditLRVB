# include "kahan_summation.h"
# include "gtest/gtest.h"

TEST(kahan_sum, is_correct) {

  KahanAccumulator<double> kahan_accum;

  kahan_accum.Add(1.0);
  kahan_accum.Add(2.0);
  kahan_accum.Add(3.0);

  EXPECT_DOUBLE_EQ(kahan_accum.value, 6.0);

  kahan_accum.Reset();

  kahan_accum.Add(1.0);
  kahan_accum.Add(2.0);

  EXPECT_DOUBLE_EQ(kahan_accum.value, 3.0);

};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
