#set page(numbering: "1")

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)


#align(center,text(14pt)[
  *Diffusion Model* \
  CMSC 25025 Final Project\
  Akash Piya
])



= Question 1
As given in the paper, $q_(t|t-1)(x_t|x_(t-1)) = cal(N)(sqrt(alpha_t)x_(t-1),(1-alpha_t)I_d)$. Through the reparameterization trick, we can write this as
$
q_(t|t-1)(x_t|x_(t-1)) = sqrt(alpha_t)x_(t-1) + epsilon_1 sqrt((1-alpha_t))
$

Writing out a few more terms in a similar manner:
$
q_(t-1|t-2) (x_(t-1)|x_(t-2)) = sqrt(alpha_(t-1))x_(t-2) + epsilon_2 sqrt((1-alpha_(t-1))) \
=> q_(t|t-2)(x_t|x_(t-2)) = sqrt(alpha_t)sqrt(alpha_(t-1))x_(t-2) +sqrt(alpha_t)sqrt(1-alpha_(t-1)) epsilon_2 + sqrt(1-alpha_t) epsilon_1 \

q_(t-2|t-3)(x_(t-2)|x_(t-3)) = sqrt(alpha_(t-2))x_(t-3) +epsilon_3sqrt(1-alpha_(t-2)) \
=> q_(t|t-3) (x_t|x_(t-3)) = sqrt(alpha_t alpha_(t-1) alpha_(t-2))x_(t-3) +sqrt(alpha_t alpha_(t-1) (1-alpha_(t-2)))epsilon_3+sqrt(alpha_t (1-alpha_(t-1)))epsilon_2 + sqrt(1-alpha_t) epsilon_1
$

where $epsilon_i in cal(N)(0,I)$.
In this form, a clear patterms forms for $q_(t|0)$ (with the substitution that $overline(alpha)_t = product_(s=1)^(t) alpha_s$). Generalizing this pattern starting at $x_0$:
$
  q_(t|0)(x_t|x_0) = sqrt(overline(alpha_t))x_0 + sum_(i=1)^(t) beta_i epsilon_i "where " beta_i = sqrt((1-alpha_(t-i+1))product_(j=0)^(i-2) alpha_(t-j))
$
One can view this expression a "reverse" reparameterization trick. The summation term alone can be interpretated as the sum of several standard normal distributions with variance $beta_i^2$ and mean $0$. Since the sum of two normal distributions is a normal distribution with mean and variance equal to the sum of the two underlying means and variance, this summation is equivalent to a single normal distribution with mean $0$ and the variance equal to the sum of all the individual variances, $beta_i^2$. Looking at the case when $t=3$ as an example, we can extract a general pattern.
$ 
sum_(i=1)^(3)beta_i^2 = alpha_t alpha_(t-1) (1-alpha_(t-2))+alpha_t (1-alpha_(t-1)) + (1-alpha_t) = 1 - alpha_t alpha_(t-1) alpha_(t-2) \
=> sum_(i=1)^(t) beta_i^2 = 1 - product_(s=1)^(t) alpha_s = 1-overline(alpha_t)
$

By this, we know that the variance of the final distribution is $1 - overline(alpha)_t$.

We can then write $q_(t|0)(x_t|x_0) = sqrt(overline(alpha_t)) x_0 + sqrt(1-overline(alpha_t)) epsilon$ where $epsilon tilde cal(N)(0,I_d)$ which implies that $q_(t|0) tilde cal(N) (sqrt(overline(alpha_t))x_0,(1-overline(alpha_t))I_d)$ by the reparameterization trick.

= Question 2
Since we are trying to maximize log-likelihood, the ELBO inequality (below) provides a tractable upperbound that we can work to maximize.
$

integral_(x_1,dots,x_T) log[(product_(s=1)^(T) p_(t-1|t)(x_(t-1)|x_t;theta)p_T (x_T))/(product_(t=1)^(T)q_(t|t-1)(x_t|x_(t-1)))] product_(t=1)^(T) q_(t|t-1)(x_t|x_(t-1)) d x_1 dots d x_T
$
Using the fact that the product of the conditional distributions $q_(t|t-1)$ is the joint distribution and basic logarithm properties, this can be rewritten as:
$
= integral_(x_1, dots, x_T) (sum_(t=1)^(T) log((p_(t-1|t)(x_(t-1)|x_t)) / (q_(t|t-1) (x_t|x_(t-1)))) + log(p_T (x_T)) )q(x_1, dots, x_T|x_0 ) d x_1, dots, d x_T
$

Note that the integral is over all $x_i$ yet if we expand the sum, each term is dependent on only two $x_i$'s so we can integrate over the rest of the marginal variables.

$
= integral_(x_T) log (p_T (x_T)) q_T (x_T|x_0) d x + sum_(t=1)^T integral_(x_(t-1), x_t) log((p_(t-1|t)(x_(t-1)|x_t)) / (q_(t|t-1) (x_t|x_(t-1)))) q(x_(t-1), x_t|x_0) d x_(t-1) d x_t
$

= Question 3
We can now define the loss that tries to maximize the ELBO upper bound by minimizing the negative of the term on the right from the equation above:

#numbered_eq(
  $
    L(theta, X_0) = sum_(t=1)^(T) integral_(x_(t-1), x_t) - log p(x_(t-1)|x_t ; theta ) q_(t-1, t|0) (x_(t-1), x_t|X_0) d x_(t-1) d x_t
  $
) <loss>

Because the forward process: $q(x_t|x_(t-1))$ follows a Gaussian, we assume that the reverse process does as well. The mean is unknown and parameterized in terms of $x_t$ and $theta$, while we assume the covariance matrix is given by $(1-alpha_t) I_d$. 
$
  p(x_(t-1)|x_t;theta) = C exp[-1/2(x_(t-1) - mu(x_t, t\;theta))^T ((1-alpha_t)I_d)^(-1)(x_(t-1) - mu(x_t, t\;theta))]
$
where $C$ is a normalization constant that we can ignore. Of note in this formulation is that the covariance matrix is diagonal with identical entries, $(1 - alpha_t)$. Hence this matrix scales vectors by a constant and can be treated as a scalar rather than a matrix:
$
  -log p(x_(t-1)|x_t;theta) = 1/ (2(1-alpha_t)) |x_(t-1) - mu(x_t, t\;theta)|^2 + C
$
Plugging this into @loss, we get that
$
  L(theta, X_0) = sum_(t=1)^T integral_(x_(t-1),x_t) (|x_(t-1) - mu(x_t, t\; theta)|^2) / (2(1-alpha_t)) q_(t-1, t|0) (x_(t-1), x_t|X_0) d x_(t-1) d x_(t) + C
$
where $C$ is again independent of $theta$. The term inside the integral looks likes an expectation function where $x_t$ and $x_(t-1)$ are drawn from the distribution $q_(t-1, t|0)(x_(t-1), x_t|X_0)$. The loss then takes the following form:
$
  L(theta, X_0) = sum_(t=1)^T E_(q_(t-1,t|0))[(|X_(t-1) - mu(X_t, t\;theta)|^2)/(2(1-alpha_t))|X_0] + C
$
= Question 4
Note that because $q$ is Markov, $q_(t|t-1,0)(x_t|x_(t-1), x_0) = q_(t|t-1) (x_t|x_(t-1))$ and that  $q_(t-1, t|0)(x_(t-1), x_t|x_0) = q_(t|t-1, 0)(x_t|x_(t-1), x_0) q_(t-1|0)(x_(t-1)|x_0)$. . This last term $q_(t-1|0)$ has a distribution specified in Question 1 that will be normal. Given this value, we can run the forward process on $x_(t-1)$ to get a distribution over $x_(t)$. Using this, we can calculate $q_(t-1, t|0)$ for all values of $x_(t-1) "and" x_t$ which we can substitute above.

= Question 5
By Bayes Theorem, 
$
  q(x_(t-1)|x_t, x_0) = (q(x_t|x_(t-1), x_0) q(x_(t-1)|x_0) )/(q(x_(t)|x_0))
$
We know that each of these terms can be written as some normal distribution and multiplied together:
$
  q(x_(t-1)|x_(t), x_0) = C exp(-1/2 (|x_t - sqrt(alpha_t) x_(t-1)|^2) / (1-alpha_t)) \
  q(x_(t-1)|x_0) = D exp (-1/2 (|x_(t-1) - sqrt(overline(alpha)_(t-1))x_0|^2) / (1-overline(alpha)_(t-1))) \
  q(x_(t)|x_0) = E exp (-1/2 (|x_(t) - sqrt(overline(alpha)_t) x_0|^2) / (1-overline(alpha)_t))
$
where $C prop 1 / sqrt(1-alpha_t), D prop 1 / sqrt(1-overline(alpha)_(t-1)), E prop 1/ sqrt(1-overline(alpha)_t)$ are all the normalization factors. Plugging these into the initial equation:

$
  q(x_(t-1)|x_(t), x_0) = (C D) / (E) exp(-1/2 [(|x_t - sqrt(alpha)_t x_(t-1)|^2) / (1 - alpha_t) + (|x_(t-1) - sqrt(overline(alpha)_(t-1)) x_0|^2) / (1 - overline(alpha)_(t-1)) - (|x_t - sqrt(overline(alpha)_t) x_0|^2) / (1-overline(alpha)_t)])
$
Because the distribution we are looking for is over $x_(t-1)$, then any other term above can be grouped into a constant that depends on $x_t$ and $x_0$.

$
= (C D) / E exp(-1/2 [(x_t^2 - 2sqrt(alpha_t) x_t x_(t-1) + alpha_t x_(t-1)^2) / (1-alpha_t) + (x_(t-1)^2 - 2 sqrt(overline(alpha)_(t-1))x_0 x_(t-1) + overline(alpha)_(t-1) x_0^2) / (1 - overline(alpha)_(t-1)) + C(x_t, x_0)]) \
= (C D) / E exp(-1/2 [x_(t-1)^2 (alpha_t / (1 - alpha_t ) + 1 / (1 - overline(alpha)_(t-1))) - x_(t-1) ((2 sqrt(alpha)_t x_t ) / (1 - alpha_t) + (2 sqrt(overline(alpha)_(t-1))x_0) / (1-overline(alpha)_(t-1))) + C(x_t, x_0)])
$
We can complete the square within the exponent:

$
a x^2 + b x = a(x + b / (2 a))^2 - b^2 / (4 a)
$
The $b^2 / (4a)$ term can be lumped with $C$. Making the appropriate substitutions:
$
a = (alpha_t / (1-alpha_t) + 1 / (1 - overline(alpha)_(t-1))) = (1 - overline(alpha)_t) / ((1-alpha_t)(1-overline(alpha)_(t-1))) \
b = -2 ((sqrt(alpha_t) x_t (1-overline(alpha)_(t-1)) + sqrt(overline(alpha)_(t-1))x_0(1-alpha_t)) / ((1-alpha_t)(1-overline(alpha)_(t-1)))) \
b / (2a) = - ((sqrt(alpha_t) x_t (1-overline(alpha)_(t-1)) + sqrt(overline(alpha)_(t-1))x_0(1-alpha_t)) / (1-overline(alpha)_t))
$

In this formulation, we have that
$
q_(t-1|t, 0) = (C D) / E exp(-1 / 2 [(1-overline(alpha)_t) / ((1-alpha_t)(1-overline(alpha)_(t-1))) (x - (sqrt(alpha)_t x_t (1-overline(alpha)_(t-1)) + sqrt(overline(alpha)_(t-1))x_0 (1-alpha_t))/(1 - overline(alpha)_t))^2+C(x_t, x_9)])
$
We know that this function is well-normalized. The $+C$ term above can be remove from the exponential and amalgamated with $(C D) / E$. Regardless, the expression above is a gaussian with mean $tilde(mu)_t$ and variance $rho_t$ where:
$
  tilde(mu)_t = (sqrt(alpha)_t x_t (1-overline(alpha)_(t-1)) + sqrt(overline(alpha)_(t-1))x_0 (1-alpha_t))/(1 - overline(alpha)_t) \
  rho_t = ((1 - alpha_t) (1-overline(alpha)_(t-1))) / (1-overline(alpha)_t)
$
= Question 6
We now examine a single term in the loss function (Question 3).
Given the initial expression:
$
  integral (|x_(t-1) - mu(x_t, t\;theta)|^2) / (2(1-alpha_t)) q_(t-1, t|0) (x_(t-1), x_t|X_0) d x_(t-1) d x_t
$
We can use the fact that $p(a, b) = p(a|b) p(b)$ on $q$.
$
  = integral (|x_(t-1) - mu(x_t, t\;theta)|^2) / (2(1-alpha_t)) q_(t-1|t, 0) (x_(t-1)|x_t, X_0) q_(t|0) (x_(t)|X_0) d x_(t-1) d x_t
$
By Question 5, we know that $q_(t-1|t, 0)$ is Gaussian so defining $tilde(mu)_t = ((1- alpha_t) sqrt(overline(alpha)_(t-1)) x_0 + (1-overline(alpha)_(t-1))sqrt(alpha_t) x_t) / (1- overline(alpha)_t)$ and $rho_t = ((1-alpha_t)(1-overline(a)_(t-1))) / (1-overline(alpha)_t)$
$
  = integral (|x_(t-1) - mu(x_t, t\;theta)|^2) / (2(1-alpha_t)) C exp[-1/2 (|x_(t-1) - tilde(mu)_t|^2)/ rho_t] q_(t|0) (x_(t)|X_0) d x_(t-1) d x_t
$
where $C = 1 / sqrt(2 pi rho_t)$. We can integrate over $x_(t-1)$. 
#numbered_eq(
  $
    = C / (2 (1-alpha_t)) integral integral_(-infinity)^(infinity)|x_(t-1) - mu(x_t, t\; theta)|^2 exp[-1/2 (|x_(t-1) - tilde(mu)_t|^2) / (rho_t)] d x_(t-1) q_(t|0) (x_t|X_0) d x_t
  $
) <init>
Let's make the substitution that $y = x_(t-1) - tilde(mu)_t$. The integral in question then becomes:
$
  = integral_(-infinity)^(infinity) |y + tilde(mu)_t - mu_t|^2 exp(-1/(2 rho_t) |y|^2) d y
$
#numbered_eq(
$
  = integral_(-infinity)^(infinity) |y|^2 exp(-1/(2 rho_t) |y|^2) d y + integral_(-infinity)^(infinity) 2y(tilde(mu)_t - mu_t) exp(-1/(2 rho_t) |y|^2) d y \ + integral_(-infinity)^(infinity) (tilde(mu)_t - mu_t)^2 exp(-1/(2 rho_t) |y|^2) d y
$
) <longeq>
The third term is a gaussian integral if one makes the substitution that $u = y / sqrt(2 rho_t)$ which is well-known to equal $sqrt(pi)$. Hence this last term is $(tilde(mu)_t - mu_t)^2 sqrt(2 rho_t pi)$. The second term is $0$ because the exponential term is an even function and the multiplicative factor is linear, hence odd. The resulting function is odd and hence the integral equals $0$.
To calculate the first term, note that:
$
  I_l = integral_(-infinity)^(infinity) exp(-l x^2) d x=> I_l^2 = integral_(-infinity)^(infinity) integral_(-infinity)^(infinity) exp(-l(x^2 + y^2)) d y d x \
  "Setting" x = r cos(theta) "and" y = r sin(theta) "and switching to polar coordinates, which adds an" r "factor" \
  I_l^2 = integral_(0)^(2 pi) d theta integral_(0)^(infinity) r exp(-l r^2) d r = pi / l => I_l = sqrt(pi / l) 
$
Now $d / (d l) I_l$ evaluated at $l=1$ equals $integral_(-infinity)^(infinity) -x^2 exp(-x^2) d x = - sqrt(pi) / 2$. Refocusing our attention on the first term of @longeq, we can make the substitution $u = (y) / sqrt(2 rho_t)$ and then use the result we derived. The first term then equals $sqrt(2 pi rho_t^3)$. 

Therefore, @longeq becomes
$
  = sqrt(2 pi rho_t) (|tilde(mu)_t - mu_t|^2 + rho_t) 
$
Plugging this result into @init, we get 
$
  = C / (2 (1-alpha_t)) integral sqrt(2 pi rho_t) (|tilde(mu)_t - mu_t|^2 + rho_t) q_(t|0)(x_t|x_0) d x_t
$
Recall that $C$ was some normalization parameter that equals $1  / sqrt(2 pi rho_t)$. Then, @init equals
$
  integral (|tilde(mu)_t (x_t, x_0) - mu(x_t, t\;theta)|^2 + rho_t) / (2 (1 - alpha_t)) q_(t|0) (x_t|x_0) d x_t
$
But this expression can be further simplified by noting that this is an expected value sampling from $q_(t|0)$. This equals:
$
  EE_(q_(t|0)) [ (|tilde(mu)_t (x_t , x_0) - mu(x_t, t\; theta)|^2 + rho_t) / (2 (1 - alpha_t)) | X_0]
$

= Question 7
Recall that 
$
tilde(mu)_t = ((1- alpha_t) sqrt(overline(alpha)_(t-1)) x_0 + (1-overline(alpha)_(t-1))sqrt(alpha_t) x_t) / (1- overline(alpha)_t) "and"
x_0 = (x_t - sqrt(1-overline(alpha)_t) epsilon_t) / (sqrt(overline(alpha)_t))
$
We then get that 
$
  tilde(mu)_t = (((1-alpha_t) (x_t - sqrt(1-overline(alpha)_t)epsilon_t)) / (sqrt(alpha_t)) + (1- overline(alpha)_(t-1))sqrt(alpha_t)x_t)/ (1 - overline(alpha)_t) = 1 / (sqrt(alpha)_t (1 - overline(alpha)_t)) [x_t - sqrt(1 - overline(alpha)_t)epsilon_t - alpha_t x_t + alpha_t sqrt(1 - overline(alpha)_t)epsilon_t + alpha_t x_t - overline(alpha)_t x_t] \
  = 1 / (sqrt(alpha)_t (1 - overline(alpha)_t)) [x_t (1 - overline(alpha)_t) - sqrt(1-overline(alpha)_t) epsilon_t (1 - alpha_t)] = 1 / sqrt(alpha_t) [x_t - (1-alpha_t) / (sqrt(1-overline(alpha)_t)) epsilon_t]
$

= Question 8
By our work in Question 6, we know that the loss function can be written as
$
  L(theta, x_0) = sum_(t=1)^T EE_(epsilon_t) [ (|tilde(mu)(x_t, x_0) - mu(x_t, t\;theta)|^2) / (2(1-alpha_t)) | x_0]
$
If we define $mu(x_t, t\;theta) = 1 / sqrt(alpha_t) [x_t - (1 - alpha_t) / (sqrt(1-overline(alpha)_t))e_t (x_t, t\;theta)]$ and use the $tilde(mu)$ found in Question 7, the loss function simplifies to
$
  = sum_(t=1)^(T) EE_(epsilon_t) [1 / (2(1-alpha_t)) |1/sqrt(alpha_t) (1 - alpha_t)/ (sqrt(1-overline(alpha)_t)) (e_t (x_t,t\;theta) - epsilon_t) |^2] \
  L(theta, X_0) = sum_(t=1)^T EE_(epsilon_t) [(1-alpha_t) / (2alpha_t (1-overline(alpha)_t)) |epsilon_t - e_t (x_t,t\;theta)|^2]
$
Therefore our network only needs to determine the predict the noise added given some $x_t$ and time step $t$.
