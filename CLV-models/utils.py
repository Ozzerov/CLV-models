__author__ = 'Alex'

from numpy import any, asarray, sum


def check_inputs(freq, rec, age):
    freq = asarray(freq)
    rec = asarray(rec)
    age = asarray(age)

    def check_recency_is_less_than_age(tx, t):
        if any(tx > t):
            raise ValueError('Some values in recency vector are larger than T vector.',
                             'This is impossible according to the model.')

    def check_frequency_of_zero_implies_recency_of_zero(x, tx):
        ix = x == 0
        if any(tx[ix] != 0):
            raise ValueError('There exist non-zero recency values when frequency is zero.',
                             'This is impossible according to the model.')

    def check_all_frequency_values_are_integer_values(x):
        if sum((x - x.astype(int)) ** 2) != 0:
            raise ValueError('There exist non-integer values in the frequency vector.',
                             'This is impossible according to the model.')

    check_recency_is_less_than_age(rec, age)
    check_frequency_of_zero_implies_recency_of_zero(freq, rec)
    check_all_frequency_values_are_integer_values(freq)
    return freq, rec, age


