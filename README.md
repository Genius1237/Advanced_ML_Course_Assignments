# ML2-assignments
Solutions to Advanced Machine Learning Assignments 2018-19 from BITS Pilani Hyderabad Campus
# Assignment 1
## Matrix Class
Include the matrix.h file in the code which needs to make use of matrix operations by
```cpp
#include "matrix.h"
```
To run any code which uses matrix functionalities, compile them along with the `matrix.cpp` file
```sh
g++ -std=c++14 matrix.cpp <other files>.cpp
```

## Example usage of Matrix Class
```cpp
// The header file matrix.h must be included
#include "matrix.h"
#include <bits/stdc++.h>
using namespace std;
int main()
{
	//Instantiation of a Matrix object
	Matrix a(1,2), b(1,2), c(2,1), d(2,2), e(1,2);

	//Access and modify data members with usual [] notation
	a[0][0]=14; a[0][1]=23;
	b[0][0]=9; b[0][1]=12;

	//Operations like matrix addition, multiplication and transpose
	c = Transpose(a - b);
	d = Transpose(a - b) * a;
	e = a + b;

	//Output the matrix in 2D form
	cout << c << d;
	cout << e;
	return 0;
}
```
