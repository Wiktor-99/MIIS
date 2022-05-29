import cmath
import numpy as np

def count_omega(number_of_elements, index_of_element):
   return cmath.exp((-2.0 * cmath.pi * 1j * index_of_element) / number_of_elements)

def get_q_for_fft(number_of_elements, index_of_element, fft_of_odd_indexes):
   return count_omega(number_of_elements, index_of_element) * fft_of_odd_indexes[index_of_element]

def apply_fft(elements_number, fft_of_even_indexes, fft_of_odd_indexes):
    output_array = np.zeros(elements_number, dtype=np.complex_)
    half_of_elements_number = int(elements_number/2)

    for index in range(half_of_elements_number):
       q, p = get_q_for_fft(elements_number, index, fft_of_odd_indexes), fft_of_even_indexes[index]
       output_array[index] = p + q
       output_array[index + half_of_elements_number] = p - q
    return output_array

def fft(input_array):
   elements_number = int(len(input_array))
   if elements_number == 1:
      return input_array

   fft_of_even_indexes, fft_of_odd_indexes = fft(input_array[0::2]), fft(input_array[1::2])

   return apply_fft(elements_number, fft_of_even_indexes, fft_of_odd_indexes)

def fft2(f):
   m, n = f.shape

   one_dimension_array = np.reshape(f, m * n)
   fft_one_dimension = fft(one_dimension_array)
   ftt_matrix = np.reshape(fft_one_dimension, (m, n))

   return ftt_matrix, m, n

def ifft2(F, m, n):
   f, M, N = fft2(np.conj(F))
   return np.array((np.matrix(np.real(np.conj(f)))/(M*N))[:m, 0:n])

def fftshift(ftt):
   M, N = ftt.shape
   M, N = int(M), int(N)
   half_n, half_m = int(N/2), int(M/2)

   first_quarter = ftt[0 : half_m, 0 : half_n]
   second_quarter = ftt[half_m : M, 0: half_n]
   third_quarter = ftt[0 : half_m, half_n : N]
   fourth_quarter = ftt[half_m : M, half_n: N]

   shifted_fft = np.zeros(ftt.shape, dtype = ftt.dtype)

   shifted_fft[half_m : M, half_n : N] = first_quarter
   shifted_fft[0 : half_m, 0 : half_n] =  fourth_quarter
   shifted_fft[half_m : M, 0 : half_n] = third_quarter
   shifted_fft[0 : half_m, half_n : N]= second_quarter

   return shifted_fft
