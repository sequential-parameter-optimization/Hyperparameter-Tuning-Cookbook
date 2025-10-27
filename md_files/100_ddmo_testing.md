---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Hypothesis Testing

## A Methodological Guide to Directional Hypothesis Testing and Sample Size Determination for Paired Experiments

### 1. The Foundational Logic of Paired-Sample Analysis

The design of an experiment is the most critical determinant of its ability to yield clear, unambiguous conclusions. Among the various designs available to researchers, the paired-sample experiment holds a position of unique power and efficiency for specific research questions. This design is characterized by a structure where each observation in one sample is uniquely and non-randomly coupled with an observation in another sample. This coupling is not a matter of convenience but a deliberate methodological choice intended to control for extraneous variability. Common applications include "before-and-after" studies, where the same subjects are measured prior to and following an intervention; studies involving naturally matched pairs, such as twins or spouses; or scenarios where two different measurement techniques are applied to the same set of specimens. This inherent relationship between observations within a pair fundamentally distinguishes the design from an independent-samples approach, where the two groups are composed of unrelated individuals.

The analytical elegance of the paired design lies in its transformation of a two-sample comparison into a more straightforward one-sample problem. Rather than analyzing the distributions of the two raw measurements, $c_1$ and $c_2$, the procedure begins by calculating a single vector of difference scores for each pair, $d_i = c_{1i} - c_{2i}$. This step effectively distills the essence of the comparison into a single set of values. The entire statistical inference is then conducted on this sample of differences, focusing on a single population parameter: the mean of the paired differences, denoted as $\mu_d$. The research question is thus reframed from "Is the mean of $c_1$ different from the mean of $c_2$?" to "Is the mean of the differences, $\mu_d$, different from a specified value?" In the vast majority of cases, this specified value is zero, representing the null hypothesis of no difference between the paired measurements.

The primary statistical advantage of this transformation is a significant enhancement in statistical power, which is the ability of a test to detect a true effect. This increase in power is not arbitrary but is a direct mathematical consequence of the design's structure. The variance of the difference between two random variables, $c_1$ and $c_2$, is defined by the formula: $\operatorname{Var}(c_1 - c_2) = \operatorname{Var}(c_1) + \operatorname{Var}(c_2) - 2 \cdot \operatorname{Cov}(c_1, c_2)$. In a paired-sample design, the measurements $c_1$ and $c_2$ are derived from the same or closely matched subjects, which typically induces a positive correlation between them. This positive correlation results in a positive covariance term, $\operatorname{Cov}(c_1, c_2)$. As this term is subtracted in the variance formula, the resulting variance of the difference scores, $\operatorname{Var}(d)$, is smaller than the sum of the individual variances that would be relevant in an independent-samples design. A smaller variance leads to a smaller standard error of the mean difference, which in turn produces a larger test statistic for a given observed difference. This amplified "signal-to-noise" ratio makes it more likely that a true effect will be deemed statistically significant, thereby increasing the power of the test.

The paired design should therefore be understood not merely as a data structure but as an active experimental control strategy. Much of the "noise" or unexplained variance in an experiment stems from stable, pre-existing differences between individual subjects (e.g., genetic predispositions, baseline health, cognitive ability). By using each subject as its own control, the paired design effectively isolates the effect of the intervention from this inter-subject variability. The causal mechanism is precise: the paired structure induces a positive correlation between the two measurements. This correlation directly reduces the variance of the difference scores. The reduced variance, for a given effect size, magnifies the test statistic, which ultimately increases the statistical power of the test. This provides a much deeper justification for the design's superiority in appropriate contexts than a simple statement of its greater power.

Furthermore, the decision to employ a paired design has profound consequences for the practical and ethical dimensions of research. Because paired designs are more powerful, they often require a smaller sample size to detect an effect of a given magnitude compared to an independent-groups design. This statistical efficiency translates directly into tangible benefits. A smaller required sample size means lower research costs, as fewer participants need to be recruited, compensated, and monitored. It can also shorten the duration of a study. In clinical research, these benefits take on an ethical imperative. By requiring fewer participants, the design minimizes the number of individuals exposed to potentially ineffective or harmful experimental treatments, aligning statistical rigor with the ethical principle of minimizing risk to research subjects.

### 2. Formulating Directional Hypotheses for a Paired Experiment

The formulation of statistical hypotheses is the formal process of translating a research question into a testable claim about a population parameter. This process is governed by a strict logical framework involving two competing statements: the null hypothesis ($H_0$) and the alternative hypothesis ($H_1$). The null hypothesis represents the claim of "no effect" or the status quo. It is the default position that is assumed to be true unless sufficient evidence is presented to refute it. In many experimental contexts, $H_0$ is the proposition that the researcher aims to reject. The alternative hypothesis, conversely, embodies the research hypothesis—it is the claim that an effect exists, for which the researcher is seeking evidence. A critical requirement of this framework is that $H_0$ and $H_1$ must be mutually exclusive and exhaustive. This means that they cannot both be true simultaneously, and together they must cover all possible values of the population parameter under investigation.

#### 2.1 Defining the Parameter of Interest and the Alternative Hypothesis

For a paired-sample experiment, the statistical inference does not focus on the individual population means of the two measurements ($\mu_{c1}$ and $\mu_{c2}$). Instead, the central parameter of interest is the population mean of the paired differences, $\mu_d = \mu_{c1} - \mu_{c2}$. The user's specific research question—that $c_1$ is better than $c_2$, where "better" is defined as a smaller value—must be translated into a precise mathematical statement about this parameter. The claim $c_1$ is smaller than $c_2$ corresponds to the population-level statement $\mu_{c1} < \mu_{c2}$. By rearranging this inequality, we can express it in terms of the parameter $\mu_d$:
$\mu_{c1} - \mu_{c2} < 0$, which is equivalent to $\mu_d < 0$.

This statement, $\mu_d < 0$, becomes the alternative hypothesis ($H_1$). It is the specific, directional claim that the experiment is designed to support. The formulation is as follows:

Alternative Hypothesis, $H_1: \mu_d < 0$

This hypothesis dictates that the statistical test will be a one-tailed test, specifically a lower-tailed or left-tailed test. The term "one-tailed" signifies that the researcher is interested only in evidence indicating a difference in a particular direction (in this case, a negative difference). The entire region of rejection for the null hypothesis will be located in the extreme left tail of the test statistic's sampling distribution.

#### 2.2 Establishing the Null Hypothesis

Once the alternative hypothesis is defined, the null hypothesis is constructed as its logical complement. There are two common conventions for stating the null hypothesis in the context of a one-tailed test.

The first, and simpler, convention is the point null hypothesis. This formulation states that the mean difference is exactly zero:

Convention 1 (Point Null): $H_0: \mu_d = 0$

This form is frequently used in statistical software outputs and introductory texts because it represents the precise boundary between the null and alternative parameter spaces.

The second, and more statistically rigorous, convention is the composite null hypothesis. This formulation includes the boundary case of zero difference as well as any possible difference in the direction opposite to the alternative hypothesis:

Convention 2 (Composite Null): $H_0: \mu_d \ge 0$

This statement posits that the mean difference is either zero or positive (i.e., $c_1$ is not smaller than $c_2$). This formulation is strongly recommended because it creates a perfect and complete partition of the parameter space. Every possible true value of $\mu_d$ falls into one, and only one, of the two sets defined by $H_0: \mu_d \ge 0$ and $H_1: \mu_d < 0$. This logical completeness ensures that if the evidence leads to a rejection of $H_0$, the only remaining possibility is that $H_1$ is true, providing a more robust and defensible inference. The mathematical symbols $\ge$ and $<$ are not merely notational conveniences; they represent a fundamental logical division of all possible states of nature. Rejecting the composite null is a refutation of an entire range of possibilities, leaving the alternative hypothesis as the sole logical conclusion.

#### 2.3 The Critical Justification for a One-Tailed Test

The choice to conduct a one-tailed test is a significant methodological decision that must be justified a priori—that is, before the data are collected or analyzed. A one-tailed test offers a distinct advantage in statistical power. By concentrating the entire significance level (alpha, $\alpha$) in one tail of the distribution, the test is more sensitive to detecting an effect in the specified direction. However, this increased power comes at a substantial cost: the test is rendered completely incapable of detecting a statistically significant effect in the opposite direction, no matter how large that effect might be.

Therefore, a one-tailed test is only appropriate when there is a strong theoretical foundation or overwhelming prior evidence suggesting that an effect in the opposite direction is either impossible or of no substantive interest. For example, if a new manufacturing process is being tested that can only possibly improve efficiency or have no effect, but cannot logically make it worse, a one-tailed test might be justified. However, consider a scenario where a new drug is developed with the belief that it will be an improvement over an existing drug. Opting for a one-tailed test to maximize the chances of detecting this improvement would mean failing to test for the possibility that the new drug is actually less effective or even harmful. Such an oversight could have serious ethical and practical consequences.

This a priori commitment to a directional hypothesis is a cornerstone of the scientific method, reflecting the principle of falsifiability. Science advances by formulating specific, testable predictions and then subjecting them to empirical scrutiny. Stating a directional hypothesis before data collection is the embodiment of this process. If a researcher were to wait, observe the direction of the sample mean, and then choose the corresponding one-tailed test to achieve a significant p-value, the process would be fundamentally corrupted. This would no longer be a test of a pre-specified hypothesis but rather an exercise in post-hoc rationalization, which invalidates the statistical inference and undermines the integrity of the research enterprise.

### 3. The Paired Samples t-Test: A Technical Examination

The paired samples t-test is the statistical procedure used to evaluate the hypotheses formulated for a paired experimental design. It is a parametric test, meaning its validity relies on certain assumptions about the underlying data. Crucially, these assumptions apply not to the raw scores of the two measurements, but to the calculated difference scores ($d$) between the pairs.

#### 3.1 Core Assumptions of the Paired t-Test

The validity of the conclusions drawn from a paired t-test is contingent upon the satisfaction of four primary assumptions.

1.  **Continuous Data:** The dependent variable, and therefore the calculated differences, must be measured on a continuous scale, which includes interval and ratio levels of measurement. Examples include variables like temperature, weight, blood pressure, or test scores.
2.  **Independence of Pairs:** The observations between pairs must be independent of one another. For instance, the outcome for one subject in a before-after study should not influence the outcome for any other subject. This assumption is typically met through the use of random sampling from the population of interest.
3.  **Absence of Significant Outliers:** The set of difference scores should not contain extreme outliers. Outliers are data points that deviate markedly from the overall pattern of the data. They can exert a disproportionate influence on the sample mean and standard deviation, potentially distorting the test statistic and leading to invalid results. The presence of outliers can be effectively diagnosed by examining a boxplot of the difference scores.
4.  **Normality of Differences:** The population of difference scores from which the sample is drawn is assumed to be approximately normally distributed. The t-test is known to be fairly "robust" to violations of this assumption, especially as the sample size increases. This means that even if the data deviate moderately from a perfect normal distribution, the test can still provide valid results. This assumption can be checked visually using tools like histograms or Q-Q plots of the differences, or more formally with statistical tests such as the Shapiro-Wilk test.

The assumptions of the paired t-test are not arbitrary rules but are direct consequences of the test's mathematical derivation. The test statistic, $t = \bar{d} / (s_d / \sqrt{n})$, is constructed under the premise that the sampling distribution of the mean difference ($\bar{d}$) is normally distributed. The Central Limit Theorem ensures this will be approximately true for large samples, but for smaller samples, the normality of the underlying population of differences is a more critical condition. Furthermore, the formula relies on the sample standard deviation of the differences ($s_d$) as an estimate of the population standard deviation ($\sigma_d$). The presence of outliers can drastically inflate $s_d$, which in turn increases the denominator of the t-statistic. This artificially reduces the magnitude of the test statistic, thereby decreasing the power of the test to detect a true effect. Therefore, satisfying these assumptions is essential for ensuring that the calculated t-statistic is a reliable measure of the evidence against the null hypothesis.

#### 3.2 A Nuanced Approach to Assumption Checking

While formal statistical tests for assumptions like normality exist, a sophisticated and practical approach is warranted. Over-reliance on tests like the Shapiro-Wilk test can be misleading. The central issue is that no real-world dataset is perfectly normal. Consequently, with a large sample size, a normality test will have high power and will likely detect even trivial and inconsequential deviations from normality, leading to a rejection of the null hypothesis of normality. Conversely, with a small sample size, the test will have low power and may fail to detect substantial and problematic deviations from normality that could invalidate the t-test.

A more informative approach focuses on assessing the impact of any non-normality. Visual inspection of the distribution of differences using histograms and Q-Q plots is often more valuable than a p-value from a formal test. The primary concerns are severe skewness or the presence of very heavy tails (i.e., a high propensity for outliers), as these can most seriously affect the validity of the t-test. The robustness of the t-test, particularly for sample sizes greater than 30, means that minor to moderate deviations from normality are generally not a cause for alarm. However, if the assumptions are severely violated—for example, if the distribution of differences is extremely skewed or contains multiple significant outliers—the researcher should forgo the paired t-test in favor of a non-parametric alternative, such as the Wilcoxon Signed-Rank Test, which does not rely on the assumption of normality.

#### 3.3 Calculating the Test Statistic

The calculation of the paired t-test statistic is mathematically equivalent to performing a one-sample t-test on the vector of difference scores, testing whether their population mean is equal to zero. The procedure involves four steps:

1.  **Calculate the mean of the differences ($\bar{d}$):** This is the sample average of all the calculated difference scores.
    $$
    \bar{d} = \frac{\sum_{i=1}^{n} d_i}{n}
    $$
    where $d_i$ is the difference for the i-th pair and $n$ is the number of pairs.

2.  **Calculate the standard deviation of the differences ($s_d$):** This measures the variability of the difference scores around their mean.
    $$
    s_d = \sqrt{\frac{\sum_{i=1}^{n} (d_i - \bar{d})^2}{n-1}}
    $$

3.  **Calculate the standard error of the mean difference ($SE_{\bar{d}}$):** This is an estimate of the standard deviation of the sampling distribution of the mean difference.
    $$
    SE_{\bar{d}} = \frac{s_d}{\sqrt{n}}
    $$

4.  **Calculate the t-statistic ($t$):** This is the ratio of the observed mean difference to its standard error. It quantifies how many standard errors the observed mean difference is away from the null-hypothesized value of zero.
    $$
    t = \frac{\bar{d} - \mu_0}{SE_{\bar{d}}} = \frac{\bar{d}}{s_d / \sqrt{n}}
    $$
    where $\mu_0$ is the value from the null hypothesis, which is 0 in this case.

#### 3.4 The Decision Framework

The calculated t-statistic is then compared to the theoretical Student's t-distribution with $n-1$ degrees of freedom (df) to determine the statistical significance of the result. This comparison can be made using two complementary approaches.

*   **The Critical Value Approach:** A critical value is determined from a t-distribution table or calculator based on the chosen significance level ($\alpha$) and the degrees of freedom (df). For the lower-tailed test ($H_1: \mu_d < 0$), this will be a negative value in the left tail of the distribution (e.g., $t_{\text{critical}}$ for $\alpha=0.05$ and df=29 is -1.699). The decision rule is: if the calculated t-statistic is less than or equal to the negative critical value ($t_{\text{calculated}} \le t_{\text{critical}}$), the null hypothesis is rejected.

*   **The p-value Approach:** The p-value represents the probability of observing a sample mean difference as extreme as, or more extreme than, the one obtained in the study, assuming that the null hypothesis is true. For a lower-tailed test, the p-value is the area under the t-distribution curve to the left of the calculated t-statistic: $p = P(T_{df} \le t_{\text{calculated}})$. The decision rule is: if the p-value is less than or equal to the significance level ($p \le \alpha$), the null hypothesis is rejected. This approach is generally preferred in modern statistical practice as it provides a more nuanced measure of the strength of evidence against the null hypothesis.

### 4. A Priori Sample Size Determination for Desired Power

One of the most critical steps in the design of a rigorous quantitative study is the a priori determination of the required sample size. Conducting a study with too few subjects runs a high risk of a Type II error—failing to detect a real effect that exists in the population. Conversely, a study with too many subjects wastes resources and may needlessly expose participants to experimental conditions. A formal power analysis allows researchers to calculate the minimum sample size needed to have a high probability of detecting an effect of a specified magnitude, thereby balancing statistical rigor with practical and ethical constraints.

#### 4.1 The Four Pillars of Power Analysis

The calculation of sample size for a paired t-test is a function of four interrelated parameters. A researcher must specify values for each of these to proceed.

*   **Significance Level ($\alpha$):** This is the probability of committing a Type I error, which is the error of rejecting the null hypothesis when it is actually true (a "false positive"). This value represents the threshold for statistical significance and is conventionally set at 0.05, corresponding to a 5% risk of a false positive conclusion.
*   **Statistical Power ($1 - \beta$):** Power is the probability of correctly rejecting the null hypothesis when it is false. It is the probability that the test will detect a true effect of a specified magnitude. The desired power is typically set at 0.80 or 0.90, which means the researcher desires an 80% or 90% chance of finding a statistically significant result if the hypothesized effect truly exists in the population. The term $\beta$ represents the probability of a Type II error (a "false negative").
*   **Effect Size ($\Delta$):** This is the magnitude of the mean difference ($\mu_d$) that the study is designed to detect. It should represent the smallest difference that is considered scientifically, clinically, or practically meaningful within the research domain. This is often the most challenging parameter to specify, as it requires substantive expertise rather than purely statistical knowledge.
*   **Standard Deviation of Differences ($\sigma_d$):** This is an estimate of the population standard deviation of the paired differences. It quantifies the expected variability, or "noise," in the difference scores. An accurate estimate of $\sigma_d$ is crucial for an accurate sample size calculation.

The process of specifying these parameters and calculating the required sample size is fundamentally an exercise in formal risk management. The researcher must explicitly balance the risk of a false positive conclusion ($\alpha$) against the risk of missing a true effect ($\beta$). This decision should be informed by the real-world consequences of each type of error. For example, in the early stages of drug development, a researcher might accept a higher $\alpha$ to avoid prematurely discarding a potentially effective compound (i.e., minimizing the risk of a Type II error). In a confirmatory trial for a drug with significant side effects, a very low $\alpha$ would be required to minimize the risk of approving an ineffective drug (a Type I error).

#### 4.2 Sample Size Calculation Formula

For instructional purposes, the sample size formula can be presented using an approximation based on the standard normal (Z) distribution. This formula is generally accurate for larger sample sizes (e.g., $n > 30$). For a one-tailed test, the formula is:
$$
n = \frac{(Z_{\alpha} + Z_{\beta})^2 \sigma_d^2}{\Delta^2}
$$
Where:

*   $n$ is the required number of pairs.
*   $Z_\alpha$ is the critical Z-value corresponding to the one-tailed significance level $\alpha$. For $\alpha = 0.05$, $Z_\alpha$ is 1.645.
*   $Z_\beta$ is the critical Z-value corresponding to the desired power. For a power of 0.80 ($\beta = 0.20$), $Z_\beta$ is 0.84.
*   $\sigma_d$ is the estimated standard deviation of the differences.
*   $\Delta$ is the target effect size or the mean difference to be detected.

While this formula provides a good estimate, more precise calculations, particularly for smaller sample sizes, are performed using the non-central t-distribution. This is the method employed by statistical software packages like G\*Power and is more accurate because it accounts for the fact that the sample standard deviation is being used to estimate the population standard deviation. The underlying principle involves finding the sample size $n$ such that the critical value under the null hypothesis defines a rejection region into which the non-central t-distribution (centered on the effect size) falls with a probability of $1 - \beta$.

#### 4.3 Practical Challenges: Estimating $\sigma_d$ and $\Delta$

The primary practical hurdles in any sample size calculation are obtaining reasonable estimates for the effect size ($\Delta$) and the variability ($\sigma_d$).

*   **Estimating $\Delta$:** The choice of $\Delta$ should be driven by subject-matter expertise. The question to ask is not "What effect do I expect to find?" but rather "What is the smallest effect that would be considered important and meaningful in my field?" This ensures the study is powered to detect a difference that matters.
*   **Estimating $\sigma_d$:** Several strategies can be employed to estimate the standard deviation of the differences:
    *   **Conduct a Pilot Study:** A small-scale preliminary study can provide a direct sample estimate ($s_d$) of the variability.
    *   **Use Data from Previous Research:** Similar studies published in the literature may report the standard deviation of the differences, which can be used as an estimate.
    *   **Estimate from Individual Standard Deviations and Correlation:** This is a particularly powerful technique. If previous research provides estimates for the standard deviations of the two individual measurements ($\sigma_1$ and $\sigma_2$) and the correlation between them ($\rho$), the standard deviation of the differences can be calculated using the formula:
        $$
        \sigma_d = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
        $$

This last formula reveals a crucial design principle. The most effective way to design a more efficient experiment (i.e., one that requires a smaller sample size) is to reduce $\sigma_d$. The formula shows that $\sigma_d$ decreases as the correlation $\rho$ increases. This provides a direct, actionable insight for experimental design: researchers should strive to maximize the correlation between the paired measurements. In a before-after study, this might mean minimizing the time between measurements to reduce the influence of other factors. In a matched-pairs study, it means matching subjects on variables that are known to be strongly related to the outcome measure. This transforms the abstract statistical concept of correlation into a practical tool for creating more powerful and efficient research.

The following table serves as a practical checklist for researchers preparing to conduct a power analysis, ensuring all necessary components are considered.

| Parameter | Symbol | Description | Common Values / Source |
| :--- | :--- | :--- | :--- |
| Significance Level | $\alpha$ | Probability of a Type I error (false positive). | 0.05, 0.01 |
| Statistical Power | $1 - \beta$ | Probability of detecting a true effect. | 0.80, 0.90 |
| Directionality | Tails | Number of tails in the hypothesis test. | One-tailed (for directional $H_1$) |
| Effect Size | $\Delta$ | Minimum meaningful mean difference ($\mu_d$) to detect. | Based on domain knowledge, clinical relevance. |
| Variability | $\sigma_d$ | Standard deviation of the paired differences. | Estimated from pilot data, literature, or correlation. |

### 5. A Comprehensive Worked Example: Testing a New Fuel Additive

This section synthesizes the preceding methodological concepts into a single, step-by-step practical example. This walkthrough illustrates the complete process of statistical inquiry, from the formulation of a research question to the final interpretation of the results.

#### 5.1 The Research Scenario

A team of automotive engineers has developed a new fuel additive ($c_1$) that is hypothesized to reduce fuel consumption in passenger vehicles compared to the standard fuel ($c_2$). In this context, a "better" outcome is a smaller value for fuel consumption, typically measured in liters per 100 kilometers (L/100km). To test this hypothesis, the team plans a paired-sample experiment. They will use a fleet of cars, and each car will be driven over a standardized test course twice: once with the standard fuel and once with the new additive. This design ensures that factors specific to each car (e.g., engine wear, tire type) are controlled for, as each car serves as its own control.

#### 5.2 Step 1: Formulating the Hypotheses

*   **Research Question:** Does the new fuel additive reduce fuel consumption?
*   **Parameter of Interest:** The population mean of the paired differences in fuel consumption, $\mu_d = \mu_{\text{additive}} - \mu_{\text{standard}}$.
*   **Hypotheses:** The research question implies a directional prediction. A reduction in consumption means the value for the additive ($\mu_{\text{additive}}$) should be less than the value for the standard fuel ($\mu_{\text{standard}}$), resulting in a negative mean difference.
    *   **Null Hypothesis ($H_0$):** $\mu_d \ge 0$. (The new additive does not reduce fuel consumption or increases it.)
    *   **Alternative Hypothesis ($H_1$):** $\mu_d < 0$. (The new additive reduces fuel consumption.)
*   **Type of Test:** This setup requires a one-tailed (lower-tailed) paired t-test.

#### 5.3 Step 2: A Priori Sample Size Calculation

Before commencing the experiment, the engineers perform a power analysis to determine the number of cars required for the study.

*   **Specification of Parameters:**
    *   **Significance Level ($\alpha$):** They choose a conventional $\alpha$ of 0.05. For a one-tailed test, the corresponding $Z_\alpha$ value is 1.645.
    *   **Statistical Power ($1 - \beta$):** They desire a power of 0.80, which is a standard choice in many fields. This corresponds to $\beta = 0.20$, and the $Z_\beta$ value is 0.84.
    *   **Effect Size ($\Delta$):** After deliberation, the team decides that a reduction of 0.5 L/100km is the minimum effect size that would be commercially viable and practically meaningful. Thus, $\Delta = -0.5$.
    *   **Standard Deviation of Differences ($\sigma_d$):** Based on a small pilot study with a few vehicles, they estimate the standard deviation of the differences to be $\sigma_d = 1.2$ L/100km.
*   **Calculation (using the Z-approximation):**
    $$
    n = \frac{(Z_{\alpha} + Z_{\beta})^2 \sigma_d^2}{\Delta^2} = \frac{(1.645 + 0.84)^2 (1.2)^2}{(-0.5)^2} = \frac{(2.485)^2 (1.44)}{0.25} = \frac{(6.175)(1.44)}{0.25} \approx 35.58
    $$
*   **Conclusion:** Since the number of subjects must be an integer, the result is rounded up. The team concludes that a sample size of 36 cars is required to achieve 80% power to detect a mean reduction of 0.5 L/100km at a 0.05 significance level.

#### 5.4 Step 3: Data Collection and Presentation (Hypothetical)

The experiment is conducted with 36 cars. For each car, fuel consumption is measured with the standard fuel ($c_2$) and the new additive ($c_1$). The difference ($d = c_1 - c_2$) is calculated for each car.

#### 5.5 Step 4: Checking the Assumptions

The engineers analyze the 36 difference scores. A boxplot reveals no extreme outliers. A histogram and a Q-Q plot of the differences show a distribution that is roughly symmetric and unimodal, suggesting that the assumption of normality is reasonably met, especially given the sample size of 36, which is large enough for the Central Limit Theorem to lend robustness to the t-test.

#### 5.6 Step 5: Calculating the Test Statistic

The analysis of the hypothetical data from the 36 cars yields the following summary statistics for the difference scores:

*   Sample Mean of the Differences ($\bar{d}$): -0.65 L/100km
*   Sample Standard Deviation of the Differences ($s_d$): 1.15 L/100km

Using these values, the test statistic is calculated:

*   **Standard Error of the Mean Difference ($SE_{\bar{d}}$):**
    $$
    SE_{\bar{d}} = \frac{s_d}{\sqrt{n}} = \frac{1.15}{\sqrt{36}} = \frac{1.15}{6} \approx 0.1917
    $$
*   **t-statistic ($t$):**
    $$
    t = \frac{\bar{d}}{SE_{\bar{d}}} = \frac{-0.65}{0.1917} \approx -3.39
    $$
*   **Degrees of Freedom (df):** df = $n - 1 = 36 - 1 = 35$.

#### 5.7 Step 6: Determining the p-value

The calculated t-statistic ($t = -3.39$) with 35 degrees of freedom is used to find the p-value. Using statistical software or a t-distribution calculator for this lower-tailed test, the p-value is found to be approximately 0.0008.

#### 5.8 Step 7: Drawing a Conclusion

This final step involves making a statistical decision and interpreting it in the context of the original research question.

*   **Statistical Decision:** The calculated p-value (0.0008) is substantially smaller than the pre-specified significance level ($\alpha = 0.05$). Therefore, the decision is to reject the null hypothesis ($H_0$).
*   **Interpretation in Context:** There is strong, statistically significant evidence to conclude that the new fuel additive is effective in reducing fuel consumption. The observed average reduction in the sample was 0.65 L/100km.
*   **Confidence Interval for the Mean Difference:** To provide a more complete picture, a confidence interval for the true mean difference is calculated. Because a one-tailed test with $\alpha = 0.05$ was performed, a corresponding 90% two-sided confidence interval is appropriate (as it leaves 5% in each tail). The formula is $\bar{d} \pm t^*_{\text{critical}} \cdot SE_{\bar{d}}$, where $t^*_{\text{critical}}$ for a 90% CI with 35 df is approximately 1.69.
    Interval = $-0.65 \pm 1.69 \cdot 0.1917$
    Interval = $-0.65 \pm 0.324$
    90% Confidence Interval: [-0.974, -0.326]
*   **Final Report Conclusion:** The study provides statistically significant evidence ($t(35) = -3.39, p = 0.0008$) that the new fuel additive reduces fuel consumption. The average reduction observed in the sample of 36 cars was 0.65 L/100km. We are 90% confident that the true mean reduction in fuel consumption for the entire population of similar vehicles lies between 0.326 and 0.974 L/100km. Since this interval does not contain zero and consists entirely of negative values (indicating a reduction), the result is consistent with the rejection of the null hypothesis and provides a range of plausible magnitudes for the additive's true effect.

This worked example demonstrates the complete narrative arc of a frequentist statistical investigation. It begins with a practical problem, translates it into a formal mathematical framework of hypotheses, employs a priori reasoning to design an experiment with adequate statistical power, executes a formal analytical procedure on the collected data, and finally, translates the abstract statistical result back into a meaningful, actionable conclusion in the original context of the problem. This process highlights the role of statistics as the logical engine that drives evidence-based discovery.