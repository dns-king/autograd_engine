#include "autograd/mlp.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using namespace autograd;
    const std::vector<std::vector<double>> x{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<double> y{0, 1, 1, 0};
    MLP model(2, {4, 1}, 2026);
    constexpr int steps = 1200;
    constexpr double learning_rate = 0.12;
    std::cout << "Training XOR (deterministic seed, CPU scalar autograd)\n";
    for (int step = 0; step < steps; ++step) {
        model.zero_grad();
        auto loss = constant(0.0);
        for (std::size_t i = 0; i < x.size(); ++i) loss = loss + binary_cross_entropy(model(x[i]), y[i]);
        loss = loss / static_cast<double>(x.size());
        loss->backward();
        model.step(learning_rate);
        if (step % 100 == 0 || step == steps - 1)
            std::cout << "step " << std::setw(4) << step << "  loss " << std::fixed << std::setprecision(6) << loss->data << '\n';
    }
    std::cout << "Final predictions:\n";
    for (std::size_t i = 0; i < x.size(); ++i)
        std::cout << x[i][0] << " xor " << x[i][1] << " -> " << std::setprecision(4) << model(x[i])->data << " (target " << y[i] << ")\n";
}
