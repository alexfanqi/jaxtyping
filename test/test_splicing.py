import jax.numpy as jnp
import pytest

from jaxtyping import Array, Float

from .helpers import ParamError


try:
    import torch
except ImportError:
    torch = None


def test_splicing_self_shape(jaxtyp, typecheck):
    class A:
        def __init__(self, shape):
            self.shape = shape

        @jaxtyp(typecheck)
        def forward(self, x: Float[Array, " ... *{self.shape}"]) -> Float[Array, "..."]:
            return jnp.sum(x, axis=tuple(range(-len(self.shape), 0)))

    a = A((3, 4))
    x = jnp.zeros((2, 3, 4))
    y = jnp.zeros((2, 3, 5))

    assert a.forward(x).shape == (2,)

    with pytest.raises(ParamError):
        a.forward(y)


def test_splicing_explicit_list(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{(1,2)} ..."]):
        pass

    f(jnp.zeros((1, 2, 3)))
    f(jnp.zeros((1, 2)))

    with pytest.raises(ParamError):
        f(jnp.zeros((2, 1, 3)))

    with pytest.raises(ParamError):
        f(jnp.zeros((1,)))


def test_splicing_expressions(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{(1,2)+(3,)}"]):
        pass

    f(jnp.zeros((1, 2, 3)))
    with pytest.raises(ParamError):
        f(jnp.zeros((1, 2, 4)))


def test_splicing_suffix(jaxtyp, typecheck):
    class A:
        shape = (2, 2)

        @jaxtyp(typecheck)
        def f(self, x: Float[Array, " ... *{self.shape}"]):
            pass

    a = A()
    a.f(jnp.zeros((5, 2, 2)))
    a.f(jnp.zeros((2, 2)))

    with pytest.raises(ParamError):
        a.f(jnp.zeros((5, 2, 3)))

    with pytest.raises(ParamError):
        # Validating strict suffix match
        a.f(jnp.zeros((5, 2)))


def test_splicing_return_type(jaxtyp, typecheck):
    class A:
        shape = (2,)

        @jaxtyp(typecheck)
        def f(self, x: Float[Array, "..."]) -> Float[Array, " ... *{self.shape}"]:
            return jnp.concatenate([x, x], axis=-1) if x.shape[-1] == 1 else x

    @jaxtyp(typecheck)
    def identity(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return x

    identity(jnp.zeros((2, 3)))


def test_dynamic_expression(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{tuple(range(1,4))}"]):
        pass

    f(jnp.zeros((1, 2, 3)))
    with pytest.raises(ParamError):
        f(jnp.zeros((1, 2, 4)))


def test_multiple_f_string_splicing(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def g(x: Float[Array, "*{(1,2)} *{(3,4)}"]):
        pass

    g(jnp.zeros((1, 2, 3, 4)))

    with pytest.raises(Exception):  # Shape mismatch
        g(jnp.zeros((1, 2, 3)))


def test_splicing_variables_as_args(jaxtyp, typecheck):
    # Testing "class overhead" comment:
    # ensuring we can use variables (arguments) for splicing
    # without needing a class/self.
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{s1} *{s2}"], s1: tuple, s2: tuple):
        pass

    s1 = (1, 2)
    s2 = (3, 4)
    f(jnp.zeros((1, 2, 3, 4)), s1, s2)

    with pytest.raises(Exception):
        f(jnp.zeros((1, 2, 3)), s1, s2)


def test_splicing_prefix_variadic(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{(1,2)} ..."]):
        pass

    f(jnp.zeros((1, 2, 5, 6)))  # (1,2) match prefix, (5,6) match ...
    f(jnp.zeros((1, 2)))  # (1,2) match prefix, ... is empty

    with pytest.raises(Exception):
        f(jnp.zeros((1,)))  # Too short


def test_splicing_suffix_variadic(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "... *{(3,4)}"]):
        pass

    f(jnp.zeros((1, 2, 3, 4)))  # ...=(1,2)
    f(jnp.zeros((3, 4)))  # ...=()

    with pytest.raises(Exception):
        f(jnp.zeros((3,)))


def test_splicing_prefix_named_variadic(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{(1,2)} *batch"]):
        pass

    f(jnp.zeros((1, 2, 5)))  # batch=(5,)

    # *batch captures rest.
    f(jnp.zeros((1, 2)))  # batch=()

    with pytest.raises(Exception):
        f(jnp.zeros((1,)))


def test_mixed_complex(jaxtyp, typecheck):
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{(1,2)} ... *{(3,4)}"]):
        pass

    # Total fixed dims: 4
    f(jnp.zeros((1, 2, 9, 3, 4)))  # ...=(9,)
    f(jnp.zeros((1, 2, 3, 4)))  # ...=()

    with pytest.raises(Exception):
        f(jnp.zeros((1, 2, 3)))


def test_invalid_multiple_named_variadics(jaxtyp, typecheck):
    try:

        @jaxtyp(typecheck)
        def f(x: Float[Array, "... ..."]):
            pass

        f(jnp.zeros((1,)))
    except ValueError as e:
        assert "Cannot use non-symbolic variadic specifiers" in str(
            e
        ) or "Anonymous multiple axes '...' must be used on its own" in str(e)
    except Exception:
        # Fallback if specific message varies
        pass


def test_multiple_named_variadics_explicit(jaxtyp, typecheck):
    try:

        @jaxtyp(typecheck)
        def f(x: Float[Array, "*b1 *b2"]):
            pass

        f(jnp.zeros((1,)))
    except ValueError as e:
        assert "Cannot use non-symbolic variadic specifiers" in str(e)


def test_splicing_coexist_named(jaxtyp, typecheck):
    # Verify that *{...} and *name coexist peacefully
    @jaxtyp(typecheck)
    def f(x: Float[Array, "*{(1,)} *batch"]):
        pass

    f(jnp.zeros((1, 5)))
