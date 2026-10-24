#include "autograd/mlp.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace autograd;
namespace {
int failures = 0;
void check(bool ok, const std::string& message) { if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; } }
void near(double got, double expected, double tol, const std::string& message) { check(std::abs(got - expected) <= tol, message + " (got " + std::to_string(got) + ", expected " + std::to_string(expected) + ")"); }
void throws(const std::function<void()>& fn, const std::string& message) { try { fn(); check(false, message); } catch (const std::domain_error&) {} }
double numeric(const std::function<ValuePtr(ValuePtr)>& fn, double x) { constexpr double h = 1e-6; return (fn(make_value(x + h))->data - fn(make_value(x - h))->data) / (2 * h); }
void gradient_check(const std::function<ValuePtr(ValuePtr)>& fn, double x, const std::string& name) {
    auto v = make_value(x); auto out = fn(v); out->backward(); near(v->grad, numeric(fn, x), 2e-5, "gradient check " + name);
}
}
int main() {
    auto x = make_value(3.0, "x"); auto y = x * x + x; y->backward(); near(x->grad, 7.0, 1e-12, "reused variable x*x+x");
    auto a = make_value(2.0), b = make_value(-4.0); auto branch = a * b + a / b; branch->backward(); near(a->grad, -4.25, 1e-12, "branch gradient a"); near(b->grad, 1.875, 1e-12, "branch gradient b");
    y->backward(); near(x->grad, 7.0, 1e-12, "default backward overwrites graph gradients"); y->backward(true); near(x->grad, 14.0, 1e-12, "accumulating backward adds leaf gradient"); y->zero_grad(); near(x->grad, 0.0, 1e-12, "zero_grad traverses ancestors");
    gradient_check([](ValuePtr v) { return exp(v) * log(v) + pow(v, 3.0) / 2.0; }, 1.3, "exp/log/pow");
    gradient_check([](ValuePtr v) { return tanh(v) + sigmoid(v) - relu(v); }, 0.7, "tanh/sigmoid/relu");
    gradient_check([](ValuePtr v) { return (v * 2.0 - 1.0) / (v + 3.0); }, 0.4, "arithmetic");
    auto zero = make_value(0.0); relu(zero)->backward(); near(zero->grad, 0.0, 0.0, "ReLU derivative at zero is zero");
    throws([&] { (void)log(make_value(0.0)); }, "log(0) throws"); throws([&] { (void)div(make_value(1), make_value(0)); }, "division by zero throws"); throws([&] { (void)pow(make_value(-1), .5); }, "fractional negative power throws");
    if (failures) return 1; std::cout << "All autograd tests passed.\n"; return 0;
}
