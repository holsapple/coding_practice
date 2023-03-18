# string1 = "co(de(fight)s)"
# while "(" in string1:
#     parcount = 0
#     charcount = -1
#     startflag = 0
#     for char in string1:
#         charcount += 1
#         if char == "(" and startflag == 0:
#             parcount += 1
#             leftpar = charcount
#             startflag = 1
#         elif char == "(" and startflag == 1:
#             parcount += 1
#         elif char == ")":
#             parcount -= 1
#         if parcount == 0 and startflag == 1:
#             rightpar = charcount
#             break
#     reversecut = string1[(rightpar - 1):leftpar:-1]
#     reversecut = reversecut.replace("(", "2hU&9gg1+nm@")
#     reversecut = reversecut.replace(")", "(")
#     reversecut = reversecut.replace("2hU&9gg1+nm@", ")")
#     string1 = string1[0:leftpar] + reversecut + string1[rightpar+1::]
#
# print(string1)

# a = [1, 2, 3, 4, 5]
# b = [i for (n, i) in enumerate(a) if (n % 2 == 0)]
# c = [i for (n, i) in enumerate(a) if (n % 2 != 0)]
# print(b, sum(b))
# print(c, sum(c))

# picture = ["abc", "def", "ghi"]
# len1 = len(picture)
# len2 = len(picture[0]) + 2
# topNbot = "*" * len2
# middle = ["*" + s + "*" for s in picture]
# print([topNbot] + middle + [topNbot])

# a = [1, 2, 3, 4]
# b = [1, 9, 4, 3]
# c = [int(a[i] != b[i]) for i in range(len(a))]
# print(c)
# print(sum(c))
# if sum(c) == 2:
#     d = [n for (n, j) in enumerate(c) if c[n] == 1]
#     print(d)
#     if (a[d[0]] == b[d[1]]) and (a[d[1]] == b[d[0]]):
#         print(True)
#     else:
#         print(False)

# from collections import Counter as C
# A = [1, 2, 3, 4]
# B = [1, 2, 4, 3]
# print(C(A))
# print(C(B))
# print(C(A) == C(B))
# print(zip(A, B))
# for a, b in zip(A, B):
#     print(a, b)
# print(sum(a != b for a, b in zip(A, B)))
# print(C(A) == C(B) and sum(a != b for a, b in zip(A, B)) < 3)

# from itertools import accumulate
# a = [-1000, 0, -2, 0]
# b = []
# for n in range(len(a) - 1):
#     if a[n] >= a[n+1]:
#         b.append(a[n] - a[n+1] + 1)
#         a[n+1] += a[n] - a[n+1] + 1
# c = list(accumulate(b))
# print(b)
# print(c)
# print(sum(b))

# from collections import Counter
# a = "aacbb"
# b = Counter(a)
# c = sum([1 for item, n in b.items() if n % 2 != 0])
# print(c)

# a = [10, 10]
# print(max(a))
# print(min(a))

# from re import split as spl
# a = "123.45.6.789"
# b = spl('\.', a)
# c = [int(item) for item in b]
# print(b)
# print(c)

# a = [5, 3, 11, 7, 9, 10, 12]
# done = False
# jump = 2
# while not done:
#     b = [1 for k in a if k % jump == 0]
#     if sum(b) == 0:
#         print(jump)
#         done = True
#     jump += 1

# im = [[36, 0, 18, 9],
#       [27, 54, 9, 0],
#       [81, 63, 72, 45]]
# print(im)
# b = []
# for i in range(len(im) - 2):
#     c = [int((sum(im[i][j:j+3]) + sum(im[i+1][j:j+3]) + sum(im[i+2][j:j+3]))/9) for j in range(len(im[0]) - 2)]
#     b.append(c)
# print(b)

# matrix = [[True, False, False],
#           [False, True, False],
#           [False, False, False]]
# TopBot = [[False] * (len(matrix[0])+2)]
# Mid = [[False] + row + [False] for row in matrix]
# newMat = TopBot + Mid + TopBot
# print(TopBot)
# print(Mid)
# print(newMat)
# ans = []
# for i in range(len(matrix)):
#     b = [sum(newMat[i][j:j+3]) + newMat[i+1][j] + newMat[i+1][j+2] + sum(newMat[i+2][j:j+3]) for j in range(len(matrix[0]))]
#     ans.append(b)
# print(ans)

# a = 246280
# b = str(a)
# c = [int(dig) for dig in b]
# print(c)
# d = [1 for n in c if n % 2 == 1]
# if sum(d) == 0:
#     print(True)
# print(False)

# import re
# a = "@_Dw2"
# if bool(re.match("\d", a[0])):
#     print(False)
# if re.match("[a-z]", a[0]) or re.match("[A-Z]", a[0]) or re.match("\d", a[0]) or re.match("_", a[0]):
#     print(False)


# from matplotlib import pyplot as plt
# import numpy as np
#
# a = np.random.randint(-10000, 10000, size=1000)
# a = sorted(a)
# # a = [-600, -402, -400, -3, -2, 9, 99, 100, 245, 246]
# # print(a)
# y = []
# for j in range(len(a)):
#     b = sum([abs(a[k] - a[j]) for k in range(len(a))])
#     # b = sum([(np.cbrt(a[k])**2 - a[k]/a[j]**2 + a[j]**2) for k in range(len(a))])
#     y.append(b)
#     if j == 0 or b < absum:
#         absum = b
#         c = [a[j]]
#     elif b == absum:
#         c.append(a[j])
# d = min(c)
# print(d)
# plt.plot(a, y, 'k', markersize=3)  # , label='monkey activity $(t_{k},x_{k})$')
# plt.xlabel('a', fontweight='bold')
# plt.ylabel('y', fontweight='bold')
# plt.title('Sum of absolute differences', fontweight='bold')
# # plt.xlim([0, 365])
# # plt.ylim([-0.5, 2])
# # plt.legend(loc=2, prop={'size': 9}, numpoints=1)
# plt.show()


# from itertools import permutations as perm
# a = ["abc", "bef", "bcc", "bec", "bbc", "bdc"]
# # # This algorithm below doesn't work correctly.
# # atup = tuple(a)
# # for k in range(len(a)):
# #     b = list(atup)
# #     c = b.pop(k)
# #     popflag = 1
# #     while popflag:
# #         for j in range(len(b)):
# #             d = b[j]
# #             diffcount = sum([1 for p in range(len(c)) if c[p] != d[p]])
# #             if diffcount == 1:
# #                 c = b.pop(j)
# #                 break
# #             elif j == len(b) - 1:
# #                 popflag = 0
# #         if not b:
# #             print(True)
# #             popflag = 0
# # print(False)
#
# # A second attempt...
# alen = len(a)
# itemlen = len(a[0])
# apermiter = perm(a)
# for permutation in apermiter:
#     for k in range(alen - 1):
#         numdiffs = sum([1 for p in range(itemlen) if permutation[k][p] != permutation[k+1][p]])
#         if numdiffs != 1:
#             break
#         elif k == alen - 2:
#             print(True, permutation)
# print(False)


# a = np.linspace(1, 1000, 1000)
# # print(a)
# y = []
# for k in range(len(a)):
#     b = np.sqrt(a[k]) - np.sqrt(a[k] + 1)
#     y.append(b)
# plt.plot(a, y, '.k', markersize=1)
# plt.xlabel('$k$', fontweight='bold')
# plt.ylabel('$a_k = \sqrt{k} - \sqrt{k+1}$', fontweight='bold')
# plt.show()


# a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# k = 3
# if len(a) % k == 0:
#     r = len(a) // k
# else:
#     r = (len(a) // k) + 1
# print(r)
# b = [a[(k*p):(k*p+k-1)] for p in range(r)]
# print(b)
# c = [x for y in b for x in y]
# print(c)


# import re
# a = "e534"
# b = re.match("[^\d]*(\d)", a)
# c = b.group(1)
# print(c)


# a = "catyebevertbyjruucqwwmoliaa"
# b = ""
# for char in a:
#     if char not in b:
#         b += char
# print(b)
# print(len(b))
# print()
# print(set(a))
# print(len(set(a)))


# a = [2, 3, 5, 1, 6]
# k = 2
# # b = [sum(a[i:i+k]) for i in range(len(a)-k+1)]
# # print(b)
# mysum = sum(a[0:k])
# mymax = mysum
# for i in range(len(a)-k):
#     mysum = mysum + a[i+k] - a[i]
#     if mysum > mymax:
#         mymax = mysum
# print(mymax)


# n = 5
# if n < 10:
#     print(0)
# degreecount = 0
# while True:
#     degreecount += 1
#     nstring = str(n)
#     nlist = []
#     for char in nstring:
#         nlist.append(int(char))
#     n = sum(nlist)
#     if n < 10:
#         print(degreecount)
#         break


# b = 'h1'
# p = 'h3'
# coldict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
# blist = [coldict[b[0]], int(b[1])]
# plist = [coldict[p[0]], int(p[1])]
# print(blist)
# print(plist)
# if abs(blist[0] - plist[0]) == abs(blist[1] - plist[1]):
#     print(True)
# else:
#     print(False)
# print(ord('a') - 96)


# import string
# a = "fyudhrygiuhdfeis"
# leftcharcount = sum([1 for char in a if char == 'a'])
# k = 0
# while k < 25:
#     k += 1
#     rightcharcount = sum([1 for char in a if char == string.ascii_lowercase[k]])
#     if rightcharcount > leftcharcount:
#         print(False)
#         break
#     leftcharcount = rightcharcount
# print(True)


# import time
#
# def pal1(st):
#     start = time.time()
#     if st == st[::-1]:
#         return st, (time.time() - start)
#     str_len = len(st)
#     start_ind = str_len // 2
#     stop_ind = str_len - 2
#     if str_len % 2 == 1:
#         arm_len = str_len // 2
#     else:
#         arm_len = str_len // 2 - 1
#     tail_consec = sum([1 for i in range(str_len) if len(set(st[i::])) == 1])
#     tail_reflect = st[(str_len - 2*tail_consec - 1):(str_len - tail_consec - 1)] == st[(str_len - tail_consec)::]
#     for ind in range(start_ind, stop_ind + 1):
#         if st[(ind - arm_len):ind] == st[:ind:-1]:
#             if tail_consec % 2 == 1 or tail_reflect:
#                 return st[:ind] + st[ind::-1], (time.time() - start)
#             else:
#                 return st[:ind] + st[(ind - 1)::-1], (time.time() - start)
#         arm_len -= 1
#     if tail_consec == 2:
#         return st[:stop_ind] + st[::-1], (time.time() - start)
#     else:
#         return st + st[stop_ind::-1], (time.time() - start)
#
#
# def pal2(st):
#     start = time.time()
#     for i in range(len(st)):
#         s = st + st[i::-1]
#         if s == s[::-1]:
#             return s, (time.time() - start)
#
#
# st = 'dingleberry'
# for k in range(20):
#     st = st + st[(len(st) // 2)::-1]
# print('The length of the starting string is {}.'.format(len(st)))
# p1, t1 = pal1(st)
# p2, t2 = pal2(st)
# print('The length of my palindrome is {}.'.format(len(p1)))
# print('The length of the rival palindrome is {}.'.format(len(p2)))
# print('My function takes {} seconds.'.format(t1))
# print('The other function takes {} seconds.'.format(t2))


# votes = [2, 3, 5, 2]
# k = 0
# # print(sum([1 for i in range(len(votes)) if (votes[i] + k) > max([votes[j] for j in range(len(votes)) if j != i])]))
# max_votes = max(votes)
# num_max_votes = votes.count(max_votes)
# if k == 0 and num_max_votes == 1:
#     print(1)
# wins = 0
# for vote in votes:
#     if vote + k > max_votes:
#         wins += 1
# print(wins)


# a = ['-', '-', 5]
# b = set(a)
# if b == {'-'}:
#     print(True)


# char = '78'
# a = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# if char in a:
#     print(True)
# else:
#     print(False)


# s = 'aaaccccccca'
# enc_s = ''
# char = s[0]
# char_count = 1
# for item in s[1::]:
#     if item == char:
#         char_count += 1
#     else:
#         if char_count > 1:
#             enc_s += str(char_count) + char
#         else:
#             enc_s += char
#         char = item
#         char_count = 1
# if char_count > 1:
#     enc_s += str(char_count) + char
# else:
#     enc_s += char
# print(enc_s)


# cell = 'd4'
# h = ord(cell[0]) - 96
# v = int(cell[1])
# count = 0
# if v + 2 < 9 and h + 1 < 9:  # 1 o'clock test
#     count += 1
# if v + 1 < 9 and h + 2 < 9:  # 2 o'clock test
#     count += 1
# if v - 1 > 0 and h + 2 < 9:  # 4 o'clock test
#     count += 1
# if v - 2 > 0 and h + 1 < 9:  # 5 o'clock test
#     count += 1
# if v - 2 > 0 and h - 1 > 0:  # 7 o'clock test
#     count += 1
# if v - 1 > 0 and h - 2 > 0:  # 8 o'clock test
#     count += 1
# if v + 1 < 9 and h - 2 > 0:  # 10 o'clock test
#     count += 1
# if v + 2 < 9 and h - 1 > 0:  # 11 o'clock test
#     count += 1
# print(count)


# n = 152
# num_list = []
# for k in range(len(str(n))):
#     temp = list(str(n))
#     temp.pop(k)
#     print(temp)
#     string = ''
#     for item in temp:
#         string += item
#     num_list.append(int(string))
# print(num_list)
# print(max(num_list))


# import re
#
# text = 'ready, steady, go!'
# # matches = re.finditer('(\w*)\W*', text)
# matches = re.finditer('([A-Za-z]*)[^A-Za-z]*', text)
# test_len = 0
# for match in matches:
#     if len(match.group(1)) > test_len:
#         longest = match.group(1)
#         test_len = len(match.group(1))
# print(longest)


# time = "02:76"
# a = time.split(':')
# print(a)
# if int(a[0]) < 24 and int(a[1]) < 60:
#     print(True)
# print(False)


# import re
#
# inputString = '123450'
# matches = re.finditer('(\d*)\D*', inputString)
# item_sum = 0
# for match in matches:
#     if len(match.group(1)) > 0:
#         item_sum += int(match.group(1))
# print(item_sum)


# matrix = [[1, 2, 1],
#           [2, 2, 2],
#           [2, 2, 2],
#           [1, 2, 3],
#           [2, 2, 1]]
# rows = len(matrix)
# columns = len(matrix[0])
# print(rows, columns)
# mats = set((matrix[i][j], matrix[i][j+1], matrix[i+1][j], matrix[i+1][j+1]) for i in range(rows - 1) for j in range(columns - 1))
# print(mats)
# print(len(mats))


# def is_prime(n):                    # This function is not absolutely necessary.
#     if type(n) != int or n < 2:     # Its functionality is essentially superseded by the function `prime_fac()` below.
#         return False                # To use `prime_fac()` for determining whether or not an integer is prime,
#     for i in range(2, n):           # one must simply compute `len(prime_fac(n))`. If that equals 1, then the integer is prime.
#         if n % i == 0:              # Note that `is_prime()` runs faster than `prime_fac()`. If your only purpose is to determine
#             return False            # if an integer is prime, `is_prime()` is probably better for very large integers.
#     return True


# import time
#
# def prime_fac(n):
#     prime_facts = []
#     d = 2
#     while d*d <= n:
#         while n % d == 0:
#             prime_facts.append(d)
#             n //= d
#         d += 1
#     if n > 1:
#         prime_facts.append(n)
#     return prime_facts
#
#
# product = 700
#
# start1 = time.time()
#
# if product == 0:
#     print(10)
# elif product < 10:
#     print(product)
# else:
#     prime_facts = prime_fac(product)
#     if (len(prime_facts) == 1) or (max(prime_facts) > 7):
#         print(-1)
#     else:
#         temp = list(tuple(prime_facts))
#         print('')
#         print('prime facts copy', temp)
#         print('prime facts', prime_facts)
#         print('')
#
#         digits_four = []
#         digits_six = []
#         digits_eight = []
#         digits_nine = []
#
#         eights = temp.count(2) // 3
#         if eights > 0:
#             print(eights)
#             digits_eight = [8] * eights
#             print(digits_eight)
#             temp = temp[3 * eights::]
#             print('prime facts copy', temp)
#             print('prime facts', prime_facts)
#             print('')
#
#         twos = temp.count(2)
#
#         nines = temp.count(3) // 2
#         if nines > 0:
#             print(nines)
#             digits_nine = [9] * nines
#             print(digits_nine)
#             temp = temp[0:twos] + temp[(2 * nines + twos)::]
#             print('prime facts copy', temp)
#             print('prime facts', prime_facts)
#             print('')
#
#         threes = temp.count(3)
#
#         if (twos > 0) and (threes == 1):
#             sixes = 1
#             print(sixes)
#             digits_six = [6]
#             print(digits_six)
#             if twos == 1:
#                 temp = temp[2::]
#             else:
#                 temp = [2] + temp[3::]
#             print('prime facts copy', temp)
#             print('prime facts', prime_facts)
#             print('')
#         elif twos == 2:
#             fours = 1
#             print(fours)
#             digits_four = [4]
#             print(digits_four)
#             temp = temp[2::]
#             print('prime facts copy', temp)
#             print('prime facts', prime_facts)
#             print('')
#
#         digits = sorted(digits_four + digits_six + digits_eight + digits_nine + temp)
#         print(digits)
#
#         ans_list = [digits[len(digits) - k - 1] * (10 ** k) for k in range(len(digits))]
#         print(ans_list)
#         print(sum(ans_list))
#         print('')
#
# print('My code took {} seconds to run.'.format(time.time() - start1))
# print('')
#
# start2 = time.time()
#
# # This is the top rated algorithm on codefights at the time I wrote my algorithm.
# # It's far slower than mine, and it doesn't work for any integer, i.e.,
# # it sucks for really big integers.
# if product == 0:
#     print(10)
# for i in range(2*3*4*5*6*7*8*9):
#     r = 1
#     for j in str(i):
#         r *= int(j)
#     if r == product:
#         print(i)
#         break
# print(-1)
#
# print('His code took {} seconds to run.'.format(time.time() - start2))


# names = ["doc", "doc", "image", "doc(1)", "doc"]
# new_names = []
# for name in names:
#     if name not in new_names:
#         new_names.append(name)
#     else:
#         counter = 1
#         done = False
#         while not done:
#             if name + '(' + str(counter) + ')' in new_names:
#                 counter += 1
#             else:
#                 new_names.append(name + '(' + str(counter) + ')')
#                 done = True
# print(new_names)


# code = "010010000110010101101100011011000110111100100001"
# char_list = [chr(int(code[(8 * k):(8 * k + 8)], 2)) for k in range(len(code) // 8)]
# print(''.join(char_list))


# def write_right(aa, slice_vec, row, col_start):
#     row_temp = list(tuple(aa[row]))
#     row_temp[col_start:(col_start + len(slice_vec))] = slice_vec
#     aa[row] = row_temp
#     return aa
#
#
# def write_down(aa, slice_vec, col, row_start):
#     col_temp = [list(tuple(aa[row_start + k])) for k in range(len(slice_vec))]
#     for k in range(len(slice_vec)):
#         col_temp[k][col] = slice_vec[k]
#     for k in range(len(slice_vec)):
#         aa[row_start + k] = col_temp[k]
#     return aa
#
#
# def write_left(aa, slice_vec, row, col_start):
#     row_temp = list(tuple(aa[row]))
#     slice_vec = slice_vec[::-1]
#     row_temp[(col_start - len(slice_vec) + 1):(col_start + 1)] = slice_vec
#     aa[row] = row_temp
#     return aa
#
#
# def write_up(aa, slice_vec, col, row_start):
#     col_temp = [list(tuple(aa[row_start - len(slice_vec) + 1 + k])) for k in range(len(slice_vec))]
#     slice_vec = slice_vec[::-1]
#     for k in range(len(slice_vec)):
#         col_temp[k][col] = slice_vec[k]
#     for k in range(len(slice_vec)):
#         aa[row_start - len(slice_vec) + 1 + k] = col_temp[k]
#     return aa
#
#
# n = 10
# a = [[0] * n] * n
# spread = [i for i in range(1, n * n + 1)]
# slice_lengths = [n] + sorted([(n - k) for k in range(1, n)] * 2)[::-1]
# slices = [spread[sum(slice_lengths[:k]):sum(slice_lengths[:(k + 1)])] for k in range(len(slice_lengths))]
# write_count = 0
# while write_count < len(slices):
#     rem = write_count % 4
#     quo = write_count // 4
#     if rem == 0:
#         ind = quo
#         start = quo
#         a = write_right(a, slices[write_count], ind, start)
#     elif rem == 1:
#         ind = n - (quo + 1)
#         start = quo + 1
#         a = write_down(a, slices[write_count], ind, start)
#     elif rem == 2:
#         ind = n - (quo + 1)
#         start = n - (quo + 2)
#         a = write_left(a, slices[write_count], ind, start)
#     elif rem == 3:
#         ind = quo
#         start = n - (quo + 2)
#         a = write_up(a, slices[write_count], ind, start)
#     write_count += 1
# print('')
# for g in a:
#     print(g)


# # grid = [[1, 3, 2, 5, 4, 6, 9, 8, 7],
# #         [4, 6, 5, 8, 7, 9, 3, 2, 1],
# #         [7, 9, 8, 2, 1, 3, 6, 5, 4],
# #         [9, 2, 1, 4, 3, 5, 8, 7, 6],
# #         [3, 5, 4, 7, 6, 8, 2, 1, 9],
# #         [6, 8, 7, 1, 9, 2, 5, 4, 3],
# #         [5, 7, 6, 9, 8, 1, 4, 3, 2],
# #         [2, 4, 3, 6, 5, 7, 1, 9, 8],
# #         [8, 1, 9, 3, 2, 4, 7, 6, 5]]
#
# grid = [[1, 3, 2, 5, 4, 6, 9, 2, 7],
#         [4, 6, 5, 8, 7, 9, 3, 8, 1],
#         [7, 9, 8, 2, 1, 3, 6, 5, 4],
#         [9, 2, 1, 4, 3, 5, 8, 7, 6],
#         [3, 5, 4, 7, 6, 8, 2, 1, 9],
#         [6, 8, 7, 1, 9, 2, 5, 4, 3],
#         [5, 7, 6, 9, 8, 1, 4, 3, 2],
#         [2, 4, 3, 6, 5, 7, 1, 9, 8],
#         [8, 1, 9, 3, 2, 4, 7, 6, 5]]
#
# for row in grid:
#     if len(set(row)) != 9:
#         print(False)
#
# for j in range(0, 9):
#     t = []
#     for i in range(0, 9):
#         t.append(grid[i][j])
#     if len(set(t)) != 9:
#         print(False)
#
# for d1 in range(1, 4):
#     for d2 in range(1, 4):
#         t = []
#         for i in range((d1-1)*3, d1*3):
#             for j in range((d2-1)*3, d2*3):
#                 t.append(grid[i][j])
#         if len(set(t)) != 9:
#             print(False)
#
# print(True)

