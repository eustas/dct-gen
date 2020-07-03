################################################################################
# DCT-II / DCT-III generator
#
# Based on:
#  - "A low multiplicative complexity fast recureise DCT-2 algorithm"
#    by Maxim Vashkevich and Alexander Petrovsky / arXiv / 20 Jul 2012
#  - TODO
################################################################################

N = 8
# Default mode is "symbolic" - generate pseudo-cod. Non-symbolic is used for
# debugging - it generates trasnsforms as matrices - thus it is easy to check
# that transform really does what is should.
SYMBOLIC = True

import math
import sys

################################################################################
# Base transforms / generators
################################################################################

if not SYMBOLIC:

  # Generate i-th input variable.
  # Variable is a vector of length N. Variable represents linear combination of
  # inputs after transformation.
  def makeVar(i):
    result = [0] * N
    result[i] = 1
    return result

  # Sum two variables.
  def add(x, y):
    return [x[i] + y[i] for i in range(N)]

  # Subtract two variables.
  def sub(x, y):
    return [x[i] - y[i] for i in range(N)]

  # Multiply variable by scalar.
  def mul(x, c):
    return [x[i] * c for i in range(N)]

  # Negate variable (same as mul(x, -1.0)).
  def neg(x):
    return [-x[i] for i in range(N)]

  # Shorthand for cosine, with argument multiplied by Pi.
  def C(a, b):
    return math.cos((a + 0.0) / (b + 0.0) * math.pi)

  # Shorthand for cosine, with argument multiplied by Pi.
  def C2(a, b):
    return 2.0 * math.cos((a + 0.0) / (b + 0.0) * math.pi)

  # Shorthand for cosine, with argument multiplied by Pi.
  def iC2(a, b):
    return 1.0 / C2(a, b)

  # Shorthand for sine, with argument multiplied by Pi.
  def S(a, b):
    return math.sin((a + 0.0) / (b + 0.0) * math.pi)

  # Shorthand for 1/2**0.5
  def SQ(a, b):
    return math.sqrt((a + 0.0) / (b + 0.0))

else: # SYMBOLIC

  CNTR = 0
  def makeTmp():
    global CNTR
    result = "t{:02d}".format(CNTR)
    CNTR = CNTR + 1
    return result

  def makeVar(i):
    return "i{:02d}".format(i)

  def add(x, y):
    tmp = makeTmp()
    print(tmp + " = " + x + " + " + y + ";")
    return tmp

  def sub(x, y):
    tmp = makeTmp()
    print(tmp + " = " + x + " - " + y + ";")
    return tmp

  def mul(x, c):
    tmp = makeTmp()
    print(tmp + " = " + x + " * " + c + ";")
    return tmp

  def neg(x):
    tmp = makeTmp()
    print(tmp + " = -" + x + ";")
    return tmp

  def C(a, b):
    return "c_c_" + str(a) + "_" + str(b)

  # 2.0 * math.cos((a + 0.0) / (b + 0.0) * math.pi)
  def C2(a, b):
    return "c_c2_" + str(a) + "_" + str(b)

  # 1.0 / C2(a, b)
  def iC2(a, b):
    return "c_ic2_" + str(a) + "_" + str(b)

  def S(a, b):
    return "c_s_" + str(a) + "_" + str(b)

  def SQ(a, b):
    return "c_sq_" + str(a) + "_" + str(b)

# endif SYMBOLIC

################################################################################
# Debug output.
################################################################################

# Print out variable.
def printVector(var):
  pieces = ["{:10.7f}".format(var[i]) for i in range(N)]
  print(" ".join(pieces))

# Print out transformation matrix.
def printTranform(varz):
  for j in range(N):
    printVector(varz[j])

# Calculates difference between transforms
def delta(a, b):
  return [sub(a[i], b[i]) for i in range(N)]

def dot(a, b):
  return [a[i] / b[i] for i in range(N)]

# Calculates difference between transforms
def multa(a, b):
  return [dot(a[i], b[i]) for i in range(N)]

# Generate 0-variable. Useful for debugging.
def zero():
  return [0] * N

def epsilon(x, i):
  return mul(x, SQ(1, 2)) if i == 0 else x

# DCT-II implementation based on definition.
def naiveDct2():
  varz = makeVars()
  result = [0] * N
  for k in range(N):
    var = zero()
    for n in range(N):
      var = add(var, epsilon(mul(varz[n], C(k * (2 * n + 1), 2 * N)), k))
    result[k] = mul(var, SQ(2, N))
  return result

# DCT-II implementation based on definition.
def naiveDct3():
  varz = makeVars()
  result = [0] * N
  for k in range(N):
    var = zero()
    for n in range(N):
      var = add(var, epsilon(mul(varz[n], C(n * (2 * k + 1), 2 * N)), n))
    result[k] = mul(var, SQ(2, N))
  return result

# DCT-IV implementation based on definition.
def naiveDct4():
  varz = makeVars()
  result = [0] * N
  for k in range(N):
    var = zero()
    for n in range(N):
      var = add(var, mul(varz[n], C((2 * k + 1) * (2 * n + 1), 4 * N)))
    result[k] = var
  return result

################################################################################
# Utilities
################################################################################

# Generate identity matrix. Usually this matrix is passed to DCT algorithm
# to generate "basis" vectors of the transform.
def makeVars():
  return [makeVar(i) for i in range(N)]

# Split list of variables info halves.
def split(x):
  m = len(x)
  m2 = m // 2
  return (x[0 : m2], x[m2 : m])

# Make a list of variables in a reverse order.
def reverse(varz):
  m = len(varz)
  result = [0] * m
  for i in range(m):
    result[i] = varz[m - 1 - i]
  return result

# Apply permutation
def permute(x, p):
 return [x[p[i]] for i in range(len(p))]

def transposePermutation(p):
  n = len(p)
  result = [0] * n
  for i in range(n):
    result[p[i]] = i
  return result

# See paper. Split even-odd elements.
def P(n):
  if n == 1:
    return [0]
  n2 = n // 2
  return [2 * i for i in range(n2)] + [2 * i + 1 for i in range(n2)]

# See paper. Interleave first and second half.
def Pt(n):
  return transposePermutation(P(n))

################################################################################
# Scheme 1
################################################################################

def B2(x):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  bottom = reverse(bottom)
  t = [add(top[i], bottom[i]) for i in range(n2)]
  b = [sub(top[i], bottom[i]) for i in range(n2)]
  return t + b

def iB2(x):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  t = [add(top[i], bottom[i]) for i in range(n2)]
  b = [sub(top[i], bottom[i]) for i in range(n2)]
  return t + reverse(b)

def B4(x, rn):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  rbottom = reverse(bottom)
  t = [sub(top[i], rbottom[i]) for i in range(n2)]
  b = [mul(bottom[i], C2(rn, 2 * N)) for i in range(n2)]
  top = [add(t[i], b[i]) for i in range(n2)]
  bottom = [sub(t[i], b[i]) for i in range(n2)]
  return top + bottom

def iB4(x, rn):
  n = len(x)
  n2 = n // 2
  if n == 1:
    raise "ooops"
  (top, bottom) = split(x)
  t = [add(top[i], bottom[i]) for i in range(n2)]
  b = [sub(top[i], bottom[i]) for i in range(n2)]
  bottom = [mul(b[i], iC2(rn, 2 * N)) for i in range(n2)]
  rbottom = reverse(bottom)
  top = [add(t[i], rbottom[i]) for i in range(n2)]
  return top + bottom

def P4(n):
  if n == 1:
    return [0]
  if n == 2:
    return [0, 1]
  n2 = n // 2
  result = [0] * n
  tc = 0
  bc = 0
  i = 0
  result[i] = tc; tc = tc + 1; i = i + 1
  turn = True
  while i < n - 1:
    if turn:
      result[i] = n2 + bc; bc = bc + 1; i = i + 1
      result[i] = n2 + bc; bc = bc + 1; i = i + 1
    else:
      result[i] = tc; tc = tc + 1; i = i + 1
      result[i] = tc; tc = tc + 1; i = i + 1
    turn = not turn
  result[i] = tc; tc = tc + 1; i = i + 1
  return result

def iP4(n):
  return transposePermutation(P4(n))

def d2n(x):
  n = len(x)
  if n == 1:
    return x
  y = B2(x)
  (top, bottom) = split(y)
  return permute(d2n(top) + d4n(bottom, N // 2), Pt(n))

def id2n(x):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(permute(x, P(n)))
  return iB2(id2n(top) + id4n(bottom, N // 2))

def d4n(x, rn):
  n = len(x)
  if n == 1:
    return x
  y = B4(x, rn)
  (top, bottom) = split(y)
  rn2 = rn // 2
  return permute(d4n(top, rn2) + d4n(bottom, N - rn2), P4(n))

def id4n(x, rn):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(permute(x, iP4(n)))
  rn2 = rn // 2
  y = id4n(top, rn2) + id4n(bottom, N -rn2)
  return iB4(y, rn)

################################################################################
# Scheme 2
################################################################################

# See paper.
def H(x):
  n = len(x)
  n2 = n // 2
  scale = SQ(1, 2)
  y = [mul(x[i], scale) for i in range(n)]
  top = [add(y[i], y[n - 1 - i]) for i in range(n2)]
  bottom = [sub(y[i], y[n - 1 - i]) for i in range(n2)]
  return top + bottom

# See paper. Same as sqrt(2) * H
def nH(x):
  n = len(x)
  n2 = n // 2
  scale = SQ(2, 1)
  top = [add(x[i], x[n - 1 - i]) for i in range(n2)]
  bottom = [sub(x[i], x[n - 1 - i]) for i in range(n2)]
  return top + bottom

# See paper.
def nHt(x):
  n = len(x)
  n2 = n // 2
  scale = SQ(2, 1)
  top = [add(x[i], x[n2 + i]) for i in range(n2)]
  bottom = [sub(x[n2 - 1 - i], x[n - 1 - i]) for i in range(n2)]
  return top + bottom

# See paper. Apply diagonal-cosines matrix.
def Cn(x, n):
  return [mul(x[k], C(2 * k + 1, 8 * n)) for k in range(n)]

# See paper. Apply diagonal-sines matrix.
def Sn(x, n):
  return [mul(x[k], S(2 * k + 1, 8 * n)) for k in range(n)]

# See paper. Flip sign of odd variables.
def D(x):
  if len(x) == 1:
    return [neg(x[0])]
  return [neg(x[i]) if i & 1 else x[i] for i in range(len(x))]

# See paper.
def R(x):
  n = len(x)
  n2 = n // 2
  (top, bottom) = split(x)
  lt = Cn(top, n2)
  rt = Sn(reverse(bottom), n2)
  lb = reverse(Sn(top, n2))
  rb = reverse(Cn(reverse(bottom), n2))
  top = [add(lt[i], rt[i]) for i in range(n2)]
  bottom = [sub(rb[i], lb[i]) for i in range(n2)]
  return top + D(bottom)

# See paper.
def U(x):
  n = len(x)
  n2 = n // 2
  n21 = n2 - 1
  (top, bottom) = split(x)
  bottom = D(reverse(bottom))
  y = top + bottom
  scale = SQ(1, 2)
  z = [mul(y[i + 1], scale) for i in range(n - 2)]
  (top, bottom) = split(z)
  for i in range(n21):
    y[i + 1] = add(top[i], bottom[i])
    y[n2 + i] = sub(top[i], bottom[i])
  y[n - 1] = neg(y[n - 1])
  return y

# See paper. Same as sqrt(2) * U
def nU(x):
  n = len(x)
  n2 = n // 2
  n21 = n2 - 1
  (top, bottom) = split(x)
  bottom = D(reverse(bottom))
  y = top + bottom
  scale = SQ(2, 1)
  z = y[1 : n - 1]
  (top, bottom) = split(z)
  y[0] = mul(y[0], scale)
  for i in range(n21):
    y[i + 1] = add(top[i], bottom[i])
    y[n2 + i] = sub(top[i], bottom[i])
  y[n - 1] = neg(mul(y[n - 1], scale))
  return y

# See paper. Core of recursive algorithm.
def cos2(x):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(H(x))
  return permute(cos2(top) + cos4(bottom), Pt(n))

# See paper. Core of recursive algorithm.
def ncos2(x):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(nH(x))
  return permute(ncos2(top) + ncos4(bottom), Pt(n))

# See paper. Core of recursive algorithm.
def cos4(x):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(R(x))
  return permute(U(cos2(top) + cos2(bottom)), Pt(n))

# See paper. Core of recursive algorithm.
def ncos4(x):
  n = len(x)
  if n == 1:
    return x
  (top, bottom) = split(R(x))
  return permute(nU(ncos2(top) + ncos2(bottom)), Pt(n))

################################################################################
# Adapters.
################################################################################

# Scale cos2 result to get final DCT-II transform.
def dct2():
  x = cos2(makeVars())
  scale = SQ(N, 2)
  x[0] = mul(x[0], SQ(2, 1))
  return [mul(x[i], scale) for i in range(N)]

def ndct2():
  x = ncos2(makeVars())
  return x
  #scale = SQ(N, 2)
  #return [mul(x[i], C(i, 2 * N)) for i in range(N)]

# Scale cos4 result to get final DCT-IV transform.
def dct4():
  x = cos4(makeVars())
  scale = SQ(N, 2)
  return [mul(x[i], scale) for i in range(N)]

def fdct3():
  x = id2n(makeVars())
  return x

def pik2(x):
  t00 = add(x[0], x[7])
  t01 = sub(x[0], x[7])
  t02 = add(x[3], x[4])
  t03 = sub(x[3], x[4])
  t04 = add(x[2], x[5])
  t05 = sub(x[2], x[5])
  t06 = add(x[1], x[6])
  t07 = sub(x[1], x[6])
  t08 = add(t00, t02)
  t09 = sub(t00, t02)
  t10 = add(t06, t04)
  t11 = sub(t06, t04)
  t12 = add(t07, t05)
  t13 = add(t01, t07)
  t14 = add(t05, t03)
  t15 = add(t11, t09)
  t16 = sub(t14, t13)
  t17 = mul(t15, math.cos(math.pi / 4.0))
  t18 = mul(t12, math.cos(math.pi / 4.0))
  t19 = mul(t16, math.cos(3.0 * math.pi / 8.0))
  t20 = add(t01, t18)
  t21 = sub(t01, t18)
  t22 = add(mul(t13, 0.5 / math.cos(3.0 * math.pi / 8.0)), t19)
  t23 = add(mul(t14, math.sqrt(2) * math.cos(3.0 * math.pi / 8.0)), t19)
  x[0] = add(t08, t10)
  x[1] = add(t20, t22)
  x[2] = add(t09, t17)
  x[3] = sub(t21, t23)
  x[4] = sub(t08, t10)
  x[5] = add(t21, t23)
  x[6] = sub(t09, t17)
  x[7] = sub(t20, t22)
  return x

def pik3(x):
  t00 = add(x[0], x[4])
  t01 = sub(x[0], x[4])
  t02 = add(x[6], x[2])
  t03 = sub(x[6], x[2])
  t04 = add(x[7], x[1])
  t05 = sub(x[7], x[1])
  t06 = add(x[5], x[3])
  t07 = sub(x[5], x[3])
  t08 = add(t04, t06)
  t09 = sub(t04, t06)
  t10 = add(t00, t02)
  t11 = sub(t00, t02)
  t12 = sub(t07, t05)
  t13 = mul(t12, C2(3, 8))
  t14 = add(mul(t03, SQ(2, 1)), t02)
  t15 = sub(t01, t14)
  t16 = add(t01, t14)
  t17 = add(mul(t05, 1.0 / C(3, 8)), t13)
  t18 = add(mul(t07, SQ(2, 1) * C2(3, 8)), t13)
  t19 = add(t08, t17)
  t20 = add(mul(t09, SQ(2, 1)), t19)
  t21 = sub(t18, t20)
  x[0] = add(t10, t08)
  x[1] = sub(t15, t19)
  x[2] = add(t16, t20)
  x[3] = add(t11, t21)
  x[4] = sub(t11, t21)
  x[5] = sub(t16, t20)
  x[6] = add(t15, t19)
  x[7] = sub(t10, t08)
  return x

def dctScale(u):
  eps = SQ(1, 2) if u == 0 else 1
  return SQ(2, N) * eps * C(u, 2 * N)

def pik2scaled():
  x = pik2(makeVars())
  return [mul(x[i], 1.0 / (N * dctScale(i))) for i in range(N)]

def id2nScaled():
  x = makeVars()
  x = [mul(x[i], 1.0 / (N * dctScale(i))) for i in range(N)]
  return id2n(x)

def pik3scaled():
  x = makeVars()
  x = [mul(x[i], dctScale(i)) for i in range(N)]
  return pik3(x)

def d2nScaled():
  x = d2n(makeVars())
  return [mul(x[i], dctScale(i)) for i in range(N)]

################################################################################
# Main.
################################################################################

# d2n * S = DCT2 = pik2 / (N * S)
# pik2 = d2n * S * S * N
# d2n_8 = Pt_8 * (d2n_4 + d4n_4(4)) * B2_8


#fast = id2nScaled()
#slow = naiveDct3()
#printTranform(multa(fast, slow))
#print("")

#printTranform(pik3(pik2(makeVars())))
#print("")
#printTranform(id2n(d2n(makeVars())))

def help():
  print("Usage: %s [N [T]]" % sys.argv[0])
  print("  N should be the power of 2, default is 8")
  print("  T is one of {A, a, B, b}, default is A")
  sys.exit()

def parseInt(s):
  try:
    return int(s)
  except ValueError:
    help()

if __name__== "__main__":
  if len(sys.argv) < 1 or len(sys.argv) > 3: help()
  if len(sys.argv) >= 2:
    N = parseInt(sys.argv[1])
    if (N & (N - 1)) != 0: help()
  type = 0
  if len(sys.argv) >= 3:
    typeOption = sys.argv[2]
    if len(typeOption) != 1: help()
    type = "AaBb".index(typeOption)
    if type == -1: help()
  if type == 0:
    vars = id2n(makeVars())
  elif type == 1:
    vars = d2n(makeVars())
  elif type == 2:
    vars = ncos2(makeVars())
  else:  # type == 3
    vars = cos2(makeVars())
  print("Output vector: " + str(vars))
