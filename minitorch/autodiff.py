from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol, runtime_checkable

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError('Need to implement for Task 1.1')
    vals1 = list(vals)
    vals2 = list(vals)
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    return (f(*vals1) - f(*vals2)) / (2 * epsilon)


variable_count = 1


@runtime_checkable
class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError('Need to implement for Task 1.4')
    sorted_vars = []
    visited = set()

    def is_constant(val):
        return not isinstance(val, Variable) or val.history is None

    def visit(var: Variable):
        if is_constant(var):
            return

        var_id = var.unique_id
        if var_id not in visited:
            visited.add(var_id)
            for parent in var.parents:
                visit(parent)
            sorted_vars.append(var)

    visit(variable)
    return reversed(sorted_vars)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError('Need to implement for Task 1.4')
    # 计算拓扑排序
    sorted_vars = topological_sort(variable)

    # 创建一个字典来存储每个变量的导数
    derivatives = {variable.unique_id: deriv}

    # 按照拓扑排序的顺序进行反向传播
    for var in sorted_vars:
        if var.unique_id in derivatives:
            d_output = derivatives[var.unique_id]
            if var.is_leaf():
                var.accumulate_derivative(d_output)
            else:
                for parent, d_parent in var.chain_rule(d_output):
                    if parent.unique_id in derivatives:
                        derivatives[parent.unique_id] += d_parent
                    else:
                        derivatives[parent.unique_id] = d_parent

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
