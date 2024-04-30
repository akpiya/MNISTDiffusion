#align(center,text(14pt)[
  *Diffusion Model* \
  CMSC 25025 Final Project\
  Akash Piya
])


= Question 1
As given in the paper, $q_(t|t-1)(x_t|x_(t-1)) = cal(N)(sqrt(alpha_t)x_(t-1),(1-alpha_t)I_d)$. But through the reparameterization trick, we can write the above as
$
q_(t|t-1)(x_t|x_(t-1)) = sqrt(alpha_t)x_(t-1) + epsilon_1 sqrt((1-alpha_t)) \
q_(t-1|t-2) (x_(t-1)|x_(t-2)) = sqrt(alpha_(t-1))x_(t-2) + epsilon_2 sqrt((1-alpha_(t-1))) \
=> q_(t|t-2)(x_t|x_(t-2)) = sqrt(alpha_t)sqrt(alpha_(t-1))x_(t-2) +sqrt(alpha_t)sqrt(1-alpha_(t-1)) epsilon_2 + sqrt(1-alpha_t) epsilon_1

$
where $epsilon_i in cal(N)(0,I)$. Furthermore, we have that
$
q_(t-2|t-3)(x_(t-2)|x_(t-3)) = sqrt(alpha_(t-2))x_(t-3) +epsilon_3sqrt(1-alpha_(t-2)) \
=> q_(t|t-3) (x_t|x_(t-3)) = sqrt(alpha_t alpha_(t-1) alpha_(t-2))x_(t-3) +sqrt(alpha_t alpha_(t-1) (1-alpha_(t-2)))epsilon_3+sqrt(alpha_t (1-alpha_(t-1)))epsilon_2 + sqrt(1-alpha_t) epsilon_1
$
In this form a clear expression forms for $q_(t|0)$ from the xpatterns above (with the substitution that $overline(a_t) = product_(s=1)^(t) alpha_s$):
$
  q_(t|0)(x_t|x_0) = sqrt(overline(alpha_t))x_0 + sum_(i=1)^(t) beta_i epsilon_i "where " beta_i = sqrt((1-alpha_(t-i+1))product_(j=0)^(i-2) alpha_(t-j))
$
The sum is simply the sum of several standard normal distributions with variance $beta_i^2$ which is equivalent to a single normal distribution with mean $0$ and the variance equal to the sum of all the individual variances.
$ 
sum_(i=1)^(3)beta_i^2 = alpha_t alpha_(t-1) (1-alpha_(t-2))+alpha_t (1-alpha_(t-1)) + (1-alpha_t) = 1 - alpha_t alpha_(t-1) alpha_(t-2) \
=> sum_(i=1)^(t) beta_i^2 = 1 - product_(s=1)^(t) alpha_s = 1-overline(alpha_t)
$

Hence, we can then write $q_(t|0)(x_t|x_0) = sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon$ where $epsilon tilde cal(N)(0,(1-overline(alpha_t))I_d)$ which implies that $q_(t|0) tilde cal(N) (sqrt(overline(alpha_t))x_0,(1-overline(alpha_t))I_d)$ by the reparameterization trick.

= Question 2
We are given that Eq 1 is 
$

integral_(x_1,dots,x_T) log[(product_(s=1)^(T) p_(t-1|t)(x_(t-1)|x_t;theta)p_T (x_T))/(product_(t=1)^(T)q_(t|t-1)(x_t|x_(t-1)))] product_(t=1)^(T) q_(t|t-1)(x_t|x_(t-1)) d x_1 dots d x_T

$
Letting $z = (x_1, dots, x_T)$ for a given $x_0$, we know that $p(z) = product_(t=1)^(T) q_(t|t-1) (x_t |x_(t-1))$. Hence the above expression can be viewed as an expected value function:
$

= bb(E)_(z tilde (x_1, dots x_T)) [log[(product_(s=1)^(T) p_(t-1|t) (x_(t-1)|x_t;theta) p_T(x_T))/()]]

$