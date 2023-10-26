#include <vector>
#include <tuple>


std::vector<std::tuple<int, int>> findPairs(const std::vector<double>& X, const std::vector<double>& Y) {
    
    // Initialize the vector of pairs to store the results
    std::vector<std::tuple<int, int>> pairs;

    int i = 0;
    int j = 0;
    int m = X.size();
    int n = Y.size();

    while (i < m && j < n) {
        if (i > 0 && X[i - 1] > Y[j]) {
            // Found a valid pair (i-1, j)
            pairs.emplace_back(i, j);
            j++;
        } else if (j > 0 && Y[j] > X[i]) {
            // Found a valid pair (i, j-1)
            pairs.emplace_back(i, j - 1);
            i++;
        } else if (X[i] <= Y[j]) {
            // Advance in X
            i++;
        } else {
            // Advance in Y
            j++;
        }
    }

    return pairs;
}

// int main() {
//     std::vector<double> X = {0.1, 0.3, 0.5, 0.7, 1.0};
//     std::vector<double> Y = {0.2, 0.4, 0.6, 0.8, 1.0};

//     std::vector<std::tuple<int, int>> pairs = findPairs(X, Y);

//     // Print the pairs (i, j)
//     std::cout << "Running" << std::endl;
//     for (const auto& pair : pairs) {
//         int i = std::get<0>(pair);
//         int j = std::get<1>(pair);
//         std::cout << "(" << i << ", " << j << ") ";
//     }

//     return 0;
// }
