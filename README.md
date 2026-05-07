# Project Title: Bias-Aware HR Attrition Predictor

**Student:** Greta Citterio  
**Course:** Logics for AI - Module 3 - Final Project  
**Format:** Initial Software Development Prototype  
---
## Introduction
This project is developed within the context of the "Logics for AI" course, specifically focusing on the core themes of Data Bias and Trust. As artificial intelligence systems are increasingly deployed in high-stakes domains such as Human Resources, ensuring that these systems make fair, transparent, and reliable decisions is of paramount importance. The central problem addressed in this work is *how automated predictive models*, when trained on historically imbalanced datasets, *can inherit representation biases and systematically produce unfair outcomes* for specific sensitive groups.

The primary aim of this project is to develop an Initial Software Development prototype that **predicts employee attrition risk** using a publicly available HR dataset of 1,480 employees. Rather than solely optimizing for predictive accuracy, this system is explicitly designed to *identify and measure data and algorithmic bias*. Furthermore, it addresses the fundamental requirement of **trustworthiness** by utilizing an interpretable machine learning model to ensure **explainability**.

> This project is about making sure AI doesn't just learn human prejudices. We are predicting if an employee will quit their job (attrition), but instead of just caring about how "accurate" the AI is, we care about how "fair" and "transparent" it is. If the data comes from a world where men and women are treated differently, the computer will naturally learn those "bad habits". This software spots those habits and tries to fix them.

### Code Attribution and Methodology
To ensure full transparency regarding the artefact's development, the following clarifies the distinction between custom logic developed for this project and the utilization of external libraries:
- **External Libraries** *(Tools utilized)*:
    - `pandas`: used strictly for data manipulation, ingestion (CSV reading), and exploratory data analysis (calculating representations and distributions).
    - `scikit-learn`: utilized as the mathematical engine for the predictive model. Specifically:
        - `DecisionTreeClassifier`: used to train the model. I chose this specific algorithm over others (like Random Forests or Neural Networks) because its tree-like structure inherently provides epistemic transparency, which is mathematically necessary for the explainability requirements of this project.
        - `train_test_split`: used to cleanly separate data, applying the `stratify` parameter to rigorously maintain the 16% attrition imbalance in both sets.
        - Metrics (`classification_report`, `confusion_matrix`): used for baseline global performance evaluation before moving to custom fairness metrics.
- **Custom Logic** *(Student's Contribution)*: while the learning algorithms are provided by sklearn, the following logic and implementation steps are original contributions developed for this project:
    - The formal mapping of categorical imbalances into actionable insights (Phase 1).
    - The configuration of the model (max_depth, class_weight) to prioritize Explainability over raw Accuracy (Phase 2).
    - The custom implementation of Formal Fairness Metrics (Phase 3).
---

## Phase 1 - Data Quality and Representation Bias
Before training any predictive model, it is essential to formally assess the **quality** and **representation** of the underlying data.

According to the core principles of logics, data will not be treated as absolute objective truths, but rather be evaluated thorugh:
* a *Computational View*: "data is anything that is stored in a symboric form on a medium" [Pietsch,2021]. In this script, categorical features are numerically encoded to ensure they can be formally manipulated by the Decision Tree algorithm;
* a *Representational View*: "data are representation of purported facts" [Lyon, 2016]. We assess how well the sample reflects the real-world population: our analysis reveals significant gender and class imbalances that could lead to biased inferences;
* a *Relational View*: "a datum is any object which (1) it is treated as (at least potential) evidence for one or more claims about the world and (2) it is possible to circulate it among individuals/groups" [Leonelli, 2016]. We examine the distribution of employees across departments and their income levels to understand the contextual structure of the organization;
* an *Informational View*: "data as the lowest Level of Abstraction of Information is any collection of syntactic items which can be written, read, organised or structured and analysed in their reference" [revised from Primiero, 2016]. Data is seen as "weighted language" providing evidence. Our fairness metrics and data quality checks (Completeness, Correctness, Accuracy) act as the formal weights used to verify the system's reliability.

The dataset used in this prototype is a public *HR Analytics dataset* comprising 1,480 employee records. An exploratory data analysis (EDA) was conducted to identify potential sources of bias that could skew the algorithm's decisions:
* **Target class imbalance:** the target variable (Attrition) is highly skewed (only ~16% of the workforce leaves). Without intervention, a learning algorithm might suffer from *Availability Bias*, trivially predicting "No Attrition" to maximize accuracy while completely failing its predictive purpose.
* **Gender representation bias:** the dataset exhibits a significant gender imbalance (889 male vs. 591 female employees). Under a closed-world assumption, the model might capture patterns specific to the majority group, systematically marginalizing the minority.
* **Departmental skewness:** the distribution across departments is highly uneven.
* Moving beyond simple representation, this phase mathematically evaluates the dataset's **data quality** across three formal dimensions:
	1.	*Completeness:* the absence of missing data, ensuring full coverage of the sampled entities. *During the formal audit, the dataset failed this check (57 missing values detected), identifying an epistemic gap that required active intervention.*
	2.	*Correctness:* the absence of syntactic or semantic contradictions (e.g., validating that 'Age' attributes cannot be negative).
	3.	*Accuracy:* the verification that numerical values fall within plausible real-world boundaries.

Documenting these baseline metrics guarantees the informational integrity of the artefact before any machine learning inference occurs.

---

## Phase 2 - Trustworthy Model Development
Following the data quality assessment, the next step involves developing a predictive model that strictly adheres to the principles of **Trustworthiness** and **Explainability**. In high-stakes domains like Human Resources, a black-box model (such as a deep neural network) might achieve superior predictive accuracy but fundamentally lacks transparency, making its logical inferences impossible to audit.

To ensure the system's reasoning can be formally verified, an interpretable classifier was selected (a **Decision Tree**). This model guarantees **Epistemic Transparency**: every prediction is the result of explicit, auditable IF-THEN propositions.

To address the representational flaws identified in Phase 1, the system actively implements strategies from the algorithmic fairness taxonomy:
* *Data Quality Enforcement*: before applying any algorithmic bias mitigation, the 57 incomplete records identified during the Phase 1 audit were removed. This methodological step was strictly necessary to restore the formal Completeness of the dataset and prevent epistemic uncertainty during model training.
* *Pre-processing Mitigation* (Fairness through Unawareness): the sensitive attribute (Gender) and other non-predictive identifiers are completely removed from the training set. While this prevents direct discrimination, Phase 3 will test if proxy variables still reconstruct this bias.
* *In-processing Mitigation*: to prevent the model from exploiting the 16% class imbalance, the learning algorithm is instantiated with balanced class weights. This mathematically penalizes misclassifications on the minority group, directly countering Availability Bias during the training logic.

The model's performance is evaluated using metrics robust to imbalanced datasets (Precision, Recall and F1-score), establishing a global baseline before transitioning to group-specific fairness audits.
> The decision of choosing to use a Decision Tree comes from the fact that it works like a transparent flowchart. If an AI makes a decision about a human life, we must be able to see *exactly* why it made that choice. Also, we try a naive fix here: we hide the "Gender" column from the AI, hoping that making it "blind" to gender will make it fair. 

---
## Phase 3 - Formal Fairness Evaluation and Ethical Trust Models
This phase shifts the focus from global statistical accuracy to rigorous **Algorithmic Fairness** and formal logic evaluation. we divided this audit into three steps.

### 3.1 Algorithmic Fairness Audit
In this section, we transition from global performance to group-specific auditing. We evaluate how the model behaves across sensitive groups (`gender`) using standard fairness definitions to detect potential discrimination:
* **Demographic Parity:** evaluates whether the selection rate is independent of the sensitive attribute;
* **Equal Opportunity & Equalized Odds:** assesses whether the True Positive Rate (TPR) and False Positive Rate (FPR) are mathematically balanced across groups;
* **Feature Importance:** this analysis surfaces the core decision nodes of the tree. By exposing the algorithm's internal logic, stakeholders can deduce that the system likely uses socio-economic features (like `OverTime` and `MonthlyIncome`) as *proxy variables*. A proxy variable is a seemingly neutral feature that highly correlates with a sensitive attribute; because of this hidden correlation, the algorithm can indirectly learn and reproduce the bias even when the sensitive attribute (`Gender`) is deleted. This logically proves that "fairness through unawareness" is ineffective.

### 3.2 Formal Trust and Epistemic Reliability
Through logical formalizations we assess if the system's logic can be logically "trusted". This includes computationally checking if predictions fall within a trustworthy interval, comparing model confidence against ground truth, and measuring the correction distance of biased inferences:
* **Trustworthiness Assestment:** a system’s trustworthiness can be formally assessed by checking if the observed probability of an outcome for a protected group falls within a predefined "trustworthy interval" $[L, U]$. In this HR context, we define the system as *trustworthy with respect to Gender* if the predicted attrition rate for female employees ($P(\hat{Y}=1 | Gender=Female)$) is consistent with the overall population baseline, allowing for a tolerance margin $\epsilon$. 
    
    FORMAL CONDITION: $$T(s) \leftrightarrow P(\hat{Y}=1 | G=f) \in [P(Y=1) - \epsilon, P(Y=1) + \epsilon]$$
          Where: $T(s)$ is the trustworthiness of the system; $P(Y=1)$ is the baseline attrition rate in the dataset; $\epsilon$ is the epistemic tolerance threshold (set here at 0.05): we set the epistemic tolerance threshold $\epsilon$ to 0.05, adopting the standard statistical significance level (5%) commonly used in empirical research to distinguish minor random variance from structural deviation.
* **Post-Hoc Trust:** algorithmic trust is formally modeled by comparing the system's internal confidence level ($\xi$) against the statistical expected probability derived from the Ground Truth ($\xi'$). The system derives a formal "TRUSTED" judgement only if the absolute difference is within a strict, predefined tolerance threshold ($\epsilon$);
* **Correction Distance Analysis:** approaching AI predictions as non-monotonic inferences (conclusions that can be invalidated by new evidence), this metric calculates the mathematical distance of a false positive from the decision boundary. It proxies the "informational weight" required to correct a biased inference, classifying errors as either highly correctable or deeply rooted.


### 3.3 Active Bias Mitigation (Proof of Concept)
After the formal evaluation we've observed that simply removing the sensitive attribute ("Fairness through Unawareness") is insufficient, as the model still reconstructs bias through proxy variables.

To actively mitigate this:
* we implement a Pre-processing Mitigation strategy, **Resampling**, by synthetically upsampling the minority group (Female employees) in the training data to match the majority group (Male employees) and we **forcefully correct** the historical representation bias;
* the resampling mitigation was evaluated not only on Demographic Parity, but also on Equal Opportunity and Equalized Odds, to provide a complete before/after fairness comparison across all three formal metrics.

> This is where we prove that hiding the gender from the AI doesn't stop it from being biased. To test this, we used formal math formulas from the course. We checked if the AI's internal confidence ($\xi$) matched reality ($\xi'$) and calculated how hard it would be to fix its mistakes (correction distance). Since the system failed these tests and proved to be "untrusted", we applied a real fix: we artificially balanced the number of men and women before training (resampling). This forced the AI to treat everyone equally.

### Theoretical Framework & Course Connections

The code translates theoretical principles into computational reality through five main pillars:

**1. Reasoning about Data and Epistemic Limits:**
the first phase of the code explicitly models the concept of *Representation Bias*. From a logical standpoint, a dataset is not a perfect mirror of reality, but a closed-world assumption based on historical sampling. The severe skewness discovered in Phase 1 (e.g., 889 males vs. 591 females) demonstrates that if an AI system blindly accepts data as objective truth, it will logically deduce and perpetuate historical inequalities. The Python script formally measures this epistemic gap before any inference occurs.

**2. Trust Models and Epistemic Transparency:**
trust in AI cannot be established purely on statistical accuracy; it requires the human agent to understand the machine agent's reasoning. This is why a `DecisionTreeClassifier` with a restricted depth was chosen. In logical terms, this model provides *Epistemic Transparency*: every prediction can be traced back through a series of explicit IF-THEN propositions. By extracting and printing the `feature_importances_`, the software allows a human-in-the-loop (e.g., an HR manager) to audit the system's logic, fulfilling the core prerequisite for a trustworthy AI system.

**3. Formal Bias Metrics vs. Unawareness:**
a naive approach to algorithmic fairness is 'fairness through unawareness'—simply deleting the sensitive attribute (`Gender`) from the training data. However, as the course highlights, logical reasoning within AI can easily reconstruct this information through proxy variables (e.g., Salary or Department). To combat this, Phase 3 implements rigorous mathematical definitions of fairness:
* **Demographic Parity** acts as a baseline check on representation in the output space.
* **Equal Opportunity** and **Equalized Odds** go deeper, formally testing the conditional independence of the model's errors relative to the sensitive group.
By hard-coding these metrics, the prototype proves that mathematical evaluation is required to guarantee fairness, as an algorithm can be technically accurate but formally unjust.

**4. Non-Monotonic Inference and Formal Trust:**
the final additions to the prototype model the AI's predictions as non-monotonic inferences (where conclusions can change if new information is added). By implementing the Correction Distance, the system formally measures how deeply rooted an error is. Furthermore, the post-hoc trust calculation demonstrates that algorithmic trust is not an absolute state, but a measurable distance between a black-box output and a transparent ground-truth distribution.

**5. Completeness of Fairness Auditing:**
a single fairness metric is insufficient to formally certify a system's equity.  Demographic Parity measures output-level representation, but a model can satisfy 
it while still exhibiting asymmetric error rates across groups. Only by jointly evaluating Demographic Parity, Equal Opportunity, and Equalized Odds (both before and after mitigation) can we formally verify that the correction does not simply shift the bias from one metric to another, a phenomenon known as *fairness gerrymandering*.

---

## Conclusion
This project successfully bridges the gap between predictive analytics and ethical AI within the high-stakes domain of Human Resources. By developing an interpretable prototype rather than a black-box model, we explicitly prioritized the core themes of data bias, epistemic transparency, and formal trustworthiness.

Through the development and formal audit of this software, several critical insights emerged:
* **Data is rarely neutral**: the exploratory analysis confirmed that historical HR datasets are inherently flawed, carrying structural imbalances like severe class skewness and gender disparity;
* **The illusion of unawareness**: the algorithmic audit mathematically proved that *"Fairness through Unawareness"* is logically insufficient. Simply removing a sensitive attribute does not prevent the model from reconstructing bias through proxy variables, as demonstrated by the massive discrepancy in equal opportunity;
*	**Trust is a measurable property**: algorithmic trust cannot be assumed from global accuracy. By applying the course's formal frameworks, we proved that concepts like post-hoc trust and correction distance can effectively quantify how reliable, or dangerously rooted, an AI's logic is;
*	**Fairness requires active enforcement**: the final proof-of-concept demonstrated that active pre-processing mitigation, such as resampling, is strictly necessary. Actively altering the representational data directly corrected the training logic, nearly eliminating the output disparities;
* **Active Mitigation must be evaluated holistically:** the proof-of-concept demonstrated that pre-processing mitigation via resampling reduces output disparities.  Critically, the complete before/after evaluation across all three fairness metrics (Demographic Parity, Equal Opportunity, and Equalized Odds) confirmed that the improvement is consistent and does not merely relocate bias from one criterion to another, strengthening the validity of the mitigation strategy.

**Limitations and Future Work:**

While this prototype provides a robust foundation for auditing algorithmic fairness, it is not without limitations. The dataset used is a simplified, public instance; real-world HR data introduces much deeper relational complexities. Future iterations could integrate formal epistemic logic frameworks to mathematically model the uncertainty of the human HR manager relying on the AI's predictions, or test more advanced in-processing algorithms.
This project underscores that building trustworthy AI is not a passive outcome of good coding, but requires a deliberate, formal commitment to fairness and explainability at every stage of the software development lifecycle.

> **To sum it up**: you cannot trust an AI just because it has high accuracy. Data carries the prejudices of the past, and if you just hide the sensitive information (like gender), the AI will still learn to discriminate using other clues. This project proves that we need to actively audit algorithms using math and logic, and then actively fix the data before training. Building fair AI is a deliberate choice, not an automatic feature.

## Citation and References
To develop the theoretical and computational framework of this project, the following resources were utilized:

> Anshika2301. (2021). *HR Analytics Dataset* [Dataset]. Kaggle. Retrieved from https://www.kaggle.com/datasets/anshika2301/hr-analytics-dataset
>
> Primiero, G. (2025). *Logics for AI - Block C: Logics for Trust and Data*. Course Lectures, University of Milan
>
> Manganini, C. and Primiero, G. (2024). *Reasoning With and About Bias*. Springer Nature Switzerland. Retrieved from https://www.researchgate.net/publication/388244532_Reasoning_With_and_About_Bias
>
> D'Asaro, F. A., Genco, F. A., and Primiero, G. (2025). *Checking trustworthiness of probabilistic computations*. Journal of Logic and Computation. Retrieved from https://academic.oup.com/logcom/article/35/6/exaf003/7974762
