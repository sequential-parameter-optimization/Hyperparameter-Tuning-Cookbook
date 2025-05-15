---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Python

## Data Types and Precision in Python

The float16 data type in numpy represents a half-precision floating point number. It uses 16 bits of memory, which gives it a precision of about 3 decimal digits.

The float32 data type in numpy represents a single-precision floating point number. It uses 32 bits of memory, which gives it a precision of about 7 decimal digits.
On the other hand, float64 represents a double-precision floating point number. It uses 64 bits of memory, which gives it a precision of about 15 decimal digits.

The reason float16 and float32 show fewer digits is because it has less precision due to using less memory.
The bits of memory are used to store the sign, exponent, and fraction parts of the floating point number, and with fewer bits, you can represent fewer digits accurately.

::: {#exm-float}
#### 16 versus 32 versus 64 bit

```{python}
import numpy as np

# Define a number
num = 0.123456789123456789

num_float16 = np.float16(num)
num_float32 = np.float32(num)
num_float64 = np.float64(num)

print("float16: ", num_float16) 
print("float32: ", num_float32)
print("float64: ", num_float64)
```

:::




## Recommendations


[Beginner's Guide to Python](https://wiki.python.org/moin/BeginnersGuide)

