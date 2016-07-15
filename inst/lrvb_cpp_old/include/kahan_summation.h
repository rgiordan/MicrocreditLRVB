# ifndef KAHAN_SUMMATION_H
# define KAHAN_SUMMATION_H

// Kahan summation
template <class T> class KahanAccumulator {
private:
  T correction;
  T intermediate_addend;
  T intermediate_value;

public:
  T value;

  KahanAccumulator() {
    correction = 0;
    intermediate_addend = 0;
    intermediate_value = 0;
    value = 0;
  }

  void Reset() {
    correction = 0;
    intermediate_addend = 0;
    intermediate_value = 0;
    value = 0;
  }

  T Add(T addend) {
    intermediate_addend = addend - correction;
    intermediate_value = value + intermediate_addend;
    correction = (intermediate_value - value) - intermediate_addend;
    value = intermediate_value;
  }
};

extern template class KahanAccumulator<double>;

# endif
