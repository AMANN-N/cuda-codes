// Include necessary header files
#include <iostream>   // For input/output operations
#include <math.h>     // For mathematical functions like fabs()



void add(int N, float *x, float *y)
{
    for (int i = 0; i < N; i++)
    {
        y[i] = x[i] + y[i];    // Add each element of x to corresponding element of y
    }
}

int main()
{
    int N = 1<<20;    // Set N to 2^20 (1,048,576) using bit shift operation
    float *x = new float[N];    // Dynamically allocate array x of size N
    float *y = new float[N];    // Dynamically allocate array y of size N

    // Initialize the arrays
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;    // Set all elements of x to 1.0
        y[i] = 2.0f;    // Set all elements of y to 2.0
    }

    add(N, x, y);    // Call the add function to add arrays x and y

    // Verify results and calculate error
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError += fabs(y[i] - 3.0f);    // Sum up absolute differences from expected value (3.0)
        std::cout << y[i] << std::endl;    // Print each result
    }
    
    std::cout << "Max error: " << maxError << std::endl;    // Print total error

    // Clean up dynamically allocated memory
    delete[] x;
    delete[] y;
    return 0;
}