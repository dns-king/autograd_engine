#include "autograd/value.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace autograd {
namespace {
void validate(const ValuePtr& v, const char* name) {
    if (!v) throw std::invalid_argument(std::string(name) + " must not be null");
}
void topo(const ValuePtr& v, std::unordered_set<Value*>& seen, std::vector<ValuePtr>& out) {
    if (!seen.insert(v.get()).second) return;
    for (const auto& parent : v->parents()) topo(parent, seen, out);
    out.push_back(v);
}
} // namespace

ValuePtr make_result(double data, std::vector<ValuePtr> parents) {
    auto out = make_value(data);
    out->parents_ = std::move(parents);
    return out;
}

Value::Value(double data, std::string label) : data(data), label(std::move(label)) {}
ValuePtr make_value(double data, std::string label) { return std::make_shared<Value>(data, std::move(label)); }
ValuePtr constant(double data) { return make_value(data); }

void Value::zero_grad() {
    std::unordered_set<Value*> seen;
    std::vector<ValuePtr> nodes;
    topo(shared_from_this(), seen, nodes);
    for (const auto& node : nodes) node->grad = 0.0;
}

void Value::backward(bool accumulate) {
    std::unordered_set<Value*> seen;
    std::vector<ValuePtr> nodes;
    topo(shared_from_this(), seen, nodes);
    // Reused intermediates must start clean. With accumulation we retain only
    // leaf gradients, so each call contributes exactly one fresh VJP.
    for (const auto& node : nodes)
        if (!accumulate || !node->parents().empty()) node->grad = 0.0;
    grad += 1.0; // accumulate=true adds a second vector-Jacobian product.
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
        if ((*it)->backward_) (*it)->backward_();
}

ValuePtr add(const ValuePtr& a, const ValuePtr& b) {
    validate(a, "left operand"); validate(b, "right operand");
    auto o = make_result(a->data + b->data, {a, b});
    o->backward_ = [a, b, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) { a->grad += o->grad; b->grad += o->grad; } };
    return o;
}
ValuePtr sub(const ValuePtr& a, const ValuePtr& b) { return add(a, neg(b)); }
ValuePtr mul(const ValuePtr& a, const ValuePtr& b) {
    validate(a, "left operand"); validate(b, "right operand");
    auto o = make_result(a->data * b->data, {a, b});
    o->backward_ = [a, b, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) { a->grad += b->data * o->grad; b->grad += a->data * o->grad; } };
    return o;
}
ValuePtr div(const ValuePtr& a, const ValuePtr& b) {
    validate(a, "left operand"); validate(b, "right operand");
    if (b->data == 0.0) throw std::domain_error("division by zero");
    auto o = make_result(a->data / b->data, {a, b});
    o->backward_ = [a, b, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) { a->grad += o->grad / b->data; b->grad -= o->grad * a->data / (b->data * b->data); } };
    return o;
}
ValuePtr neg(const ValuePtr& a) {
    validate(a, "operand"); auto o = make_result(-a->data, {a});
    o->backward_ = [a, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad -= o->grad; }; return o;
}
ValuePtr pow(const ValuePtr& a, double exponent) {
    validate(a, "base");
    if (a->data < 0.0 && std::floor(exponent) != exponent) throw std::domain_error("fractional power of negative base");
    if (a->data == 0.0 && exponent < 0.0) throw std::domain_error("zero to negative power");
    auto o = make_result(std::pow(a->data, exponent), {a});
    o->backward_ = [a, exponent, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad += exponent * std::pow(a->data, exponent - 1.0) * o->grad; };
    return o;
}
ValuePtr exp(const ValuePtr& a) {
    validate(a, "operand"); const double d = std::exp(a->data);
    if (!std::isfinite(d)) throw std::overflow_error("exp result is not finite");
    auto o = make_result(d, {a}); o->backward_ = [a, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad += o->data * o->grad; }; return o;
}
ValuePtr log(const ValuePtr& a) {
    validate(a, "operand"); if (a->data <= 0.0) throw std::domain_error("log input must be positive");
    auto o = make_result(std::log(a->data), {a}); o->backward_ = [a, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad += o->grad / a->data; }; return o;
}
ValuePtr tanh(const ValuePtr& a) {
    validate(a, "operand"); auto o = make_result(std::tanh(a->data), {a});
    o->backward_ = [a, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad += (1.0 - o->data * o->data) * o->grad; }; return o;
}
ValuePtr relu(const ValuePtr& a) {
    validate(a, "operand"); auto o = make_result(a->data > 0.0 ? a->data : 0.0, {a});
    o->backward_ = [a, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad += (a->data > 0.0 ? 1.0 : 0.0) * o->grad; }; return o;
}
ValuePtr sigmoid(const ValuePtr& a) {
    validate(a, "operand");
    const double d = a->data >= 0 ? 1.0 / (1.0 + std::exp(-a->data)) : std::exp(a->data) / (1.0 + std::exp(a->data));
    auto o = make_result(d, {a}); o->backward_ = [a, weak = std::weak_ptr<Value>(o)] { if (auto o = weak.lock()) a->grad += o->data * (1.0 - o->data) * o->grad; }; return o;
}
} // namespace autograd
