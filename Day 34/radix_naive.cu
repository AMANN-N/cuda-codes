#include <iostream>
#include <vector>
using namespace std;

void radix_sort_naive(vector<unsigned int>& arr) {
    int n = arr.size();
    vector<unsigned int> output(n);

    for (int bit = 0; bit < 32; bit++) 
    { 
        int idx = 0;

        for (int i = 0; i < n; i++) 
        {
            if (((arr[i] >> bit) & 1) == 0) 
            {
                output[idx++] = arr[i];
            }
        }

        for (int i = 0; i < n; i++) {
            if (((arr[i] >> bit) & 1) == 1) {
                output[idx++] = arr[i];
            }
        }

        for (int i = 0; i < n; i++) {
            arr[i] = output[i];
        }
    }
}

int main() {
    vector<unsigned int> arr = {13, 3, 2, 11, 7, 5, 1, 8};

    cout << "Original array:\n";
    for (auto x : arr) cout << x << " ";
    cout << "\n";

    radix_sort_naive(arr);

    cout << "Sorted array:\n";
    for (auto x : arr) cout << x << " ";
    cout << "\n";

    return 0;
}
