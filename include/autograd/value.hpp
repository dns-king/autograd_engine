#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace autograd {

class Value;
using ValuePtr = std::shared_ptr<Value>;

// A node owns strong references to its parents. The graph is a DAG (no child
// references), so shared ownership safely retains intermediates without cycles.
class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad = 0.0;
    std::string label;

    explicit Value(double data, std::string label = {});
    void backward(bool accumulate = false);
    void zero_grad(); // Zeros this node and every ancestor, once each.
    const std::vector<ValuePtr>& parents() const noexcept { return parents_; }

private:
    std::vector<ValuePtr> parents_;
    std::function<void()> backward_;
    friend ValuePtr make_value(double, std::string);
    friend ValuePtr make_result(double, std::vector<ValuePtr>);
    friend ValuePtr add(const ValuePtr&, const ValuePtr&);
    friend ValuePtr sub(const ValuePtr&, const ValuePtr&);
    friend ValuePtr mul(const ValuePtr&, const ValuePtr&);
    friend ValuePtr div(const ValuePtr&, const ValuePtr&);
    friend ValuePtr neg(const ValuePtr&);
    friend ValuePtr pow(const ValuePtr&, double);
    friend ValuePtr exp(const ValuePtr&);
    friend ValuePtr log(const ValuePtr&);
    friend ValuePtr tanh(const ValuePtr&);
    friend ValuePtr relu(const ValuePtr&);
    friend ValuePtr sigmoid(const ValuePtr&);
};

ValuePtr make_value(double data, std::string label = {});
ValuePtr constant(double data);
ValuePtr add(const ValuePtr& a, const ValuePtr& b);
ValuePtr sub(const ValuePtr& a, const ValuePtr& b);
ValuePtr mul(const ValuePtr& a, const ValuePtr& b);
ValuePtr div(const ValuePtr& a, const ValuePtr& b); // throws on zero divisor
ValuePtr neg(const ValuePtr& a);
ValuePtr pow(const ValuePtr& a, double exponent); // negative base requires integral exponent
ValuePtr exp(const ValuePtr& a);
ValuePtr log(const ValuePtr& a); // throws unless input > 0
ValuePtr tanh(const ValuePtr& a);
ValuePtr relu(const ValuePtr& a); // derivative at zero is defined as 0
ValuePtr sigmoid(const ValuePtr& a);

inline ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) { return add(a, b); }
inline ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) { return sub(a, b); }
inline ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) { return mul(a, b); }
inline ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) { return div(a, b); }
inline ValuePtr operator-(const ValuePtr& a) { return neg(a); }
inline ValuePtr operator+(const ValuePtr& a, double b) { return add(a, constant(b)); }
inline ValuePtr operator+(double a, const ValuePtr& b) { return constant(a) + b; }
inline ValuePtr operator-(const ValuePtr& a, double b) { return sub(a, constant(b)); }
inline ValuePtr operator-(double a, const ValuePtr& b) { return constant(a) - b; }
inline ValuePtr operator*(const ValuePtr& a, double b) { return mul(a, constant(b)); }
inline ValuePtr operator*(double a, const ValuePtr& b) { return constant(a) * b; }
inline ValuePtr operator/(const ValuePtr& a, double b) { return div(a, constant(b)); }
inline ValuePtr operator/(double a, const ValuePtr& b) { return constant(a) / b; }

} // namespace autograd
