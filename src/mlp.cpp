#include "autograd/mlp.hpp"
#include <stdexcept>

namespace autograd {
MLP::MLP(std::size_t inputs, const std::vector<std::size_t>& widths, unsigned seed) {
    if (inputs == 0 || widths.empty()) throw std::invalid_argument("MLP needs positive input and nonempty widths");
    std::mt19937 rng(seed); std::uniform_real_distribution<double> init(-1.0, 1.0);
    for (auto width : widths) {
        if (width == 0) throw std::invalid_argument("layer width must be positive");
        std::vector<Neuron> layer;
        for (std::size_t j = 0; j < width; ++j) {
            Neuron n; n.b = make_value(init(rng), "bias"); parameters_.push_back(n.b);
            for (std::size_t i = 0; i < inputs; ++i) { n.w.push_back(make_value(init(rng), "weight")); parameters_.push_back(n.w.back()); }
            layer.push_back(std::move(n));
        }
        layers_.push_back(std::move(layer)); inputs = width;
    }
}
ValuePtr MLP::operator()(const std::vector<double>& x) const {
    std::vector<ValuePtr> a; for (double v : x) a.push_back(constant(v));
    if (a.size() != layers_.front().front().w.size()) throw std::invalid_argument("input vector has wrong size");
    for (std::size_t l = 0; l < layers_.size(); ++l) {
        std::vector<ValuePtr> next;
        for (const auto& n : layers_[l]) { auto z = n.b; for (std::size_t i = 0; i < a.size(); ++i) z = z + n.w[i] * a[i]; next.push_back(l + 1 == layers_.size() ? sigmoid(z) : tanh(z)); }
        a = std::move(next);
    }
    if (a.size() != 1) throw std::logic_error("MLP output layer must contain exactly one neuron");
    return a.front();
}
void MLP::zero_grad() { for (const auto& p : parameters_) p->grad = 0.0; }
void MLP::step(double learning_rate) { if (!(learning_rate > 0.0)) throw std::invalid_argument("learning rate must be positive"); for (const auto& p : parameters_) p->data -= learning_rate * p->grad; }
ValuePtr binary_cross_entropy(const ValuePtr& p, double target) {
    if (target != 0.0 && target != 1.0) throw std::invalid_argument("binary target must be 0 or 1");
    return -(target * log(p) + (1.0 - target) * log(1.0 - p));
}
} // namespace autograd
