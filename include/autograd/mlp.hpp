#pragma once
#include "autograd/value.hpp"
#include <random>

namespace autograd {
class MLP {
public:
    MLP(std::size_t inputs, const std::vector<std::size_t>& widths, unsigned seed = 42);
    ValuePtr operator()(const std::vector<double>& x) const;
    const std::vector<ValuePtr>& parameters() const noexcept { return parameters_; }
    void zero_grad();
    void step(double learning_rate);
private:
    struct Neuron { std::vector<ValuePtr> w; ValuePtr b; };
    std::vector<std::vector<Neuron>> layers_;
    std::vector<ValuePtr> parameters_;
};
ValuePtr binary_cross_entropy(const ValuePtr& probability, double target);
} // namespace autograd
