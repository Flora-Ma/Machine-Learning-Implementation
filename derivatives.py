import sympy

J, w1, w2 = sympy.symbols('J, w1, w2')
J = w1 ** 3 + 2 * w1 * w2 + w2 ** 2
dJ_dw1 = sympy.diff(J, w1)
dJ_dw2 = sympy.diff(J, w2)
print(f'dJ_dw1={dJ_dw1}, dJ_dw2={dJ_dw2}')
print(f'at point w1=2, w2=1, dJ_dw1={dJ_dw1.subs([(w1,2), (w2,1)])}, dJ_dw2={dJ_dw2.subs([(w1,2), (w2,1)])}')
