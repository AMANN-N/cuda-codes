#include <iostream>
using namespace std;

void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] < B[j]) C[k++] = A[i++];
        else C[k++] = B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

int main() {
    int A[] = {1, 3, 5, 7};
    int B[] = {2, 4, 6, 8};
    int C[8];

    merge_sequential(A, 4, B, 4, C);

    cout << "Sequential Merged Array: ";
    for (int i = 0; i < 8; i++) cout << C[i] << " ";
    cout << endl;
    return 0;
}
