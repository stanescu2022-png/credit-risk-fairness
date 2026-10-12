# Current project structure

10 October 2026. Solid arrows show the implemented workflow. Dashed arrows show the proposed common evaluation, which is not implemented. This describes the current project; it does not authorize a refactor or new experiment.

```mermaid
flowchart TB
    A["Data preparation"] --> B["Exploratory analysis"]
    B --> C["Shared data and fixed split<br/>24,004 training / 5,996 holdout<br/>22 predictors; ID/SEX/target are metadata"]

    subgraph Original["Existing project"]
        D["Frozen sklearn models<br/>Regularized LR + full GBM"]
        E["Our existing assessment<br/>Native SAFE + performance, fairness,<br/>SHAP, robustness and uncertainty"]
        F["Completed model profiles<br/>and saved reports"]
        D --> E --> F
    end

    subgraph Article["Separate article implementation"]
        G["Training-only selection<br/>19,203 fitting / 4,801 validation"]
        H["Final GBMs<br/>Full: 22 predictors<br/>Selected: 3 predictors"]
        L["Unpenalized article LR<br/>Not estimable in this design<br/>Dependent results unavailable"]
        I["Article-based assessment<br/>DeLong, SHAP, Gini/KS,<br/>parameter tests and supporting uncertainty"]
        J["Article profiles and report<br/>Explicit missing LR results"]
        G --> H --> I --> J
        G --> L
        L --> J
    end

    C --> D
    C --> G
    F -.-> K["Not implemented<br/>Common framework decision rule<br/>and outcome/material-harm evaluation"]
    J -.-> K

    classDef common fill:#eef2f6,stroke:#64748b,color:#17212f;
    classDef existing fill:#e7f3ee,stroke:#378366,color:#17212f;
    classDef article fill:#e8f0fa,stroke:#426991,color:#17212f;
    classDef missing fill:#fff0de,stroke:#b57529,color:#17212f;
    classDef pending fill:#f7f7f7,stroke:#777,stroke-dasharray:5 5,color:#333;
    class A,B,C common;
    class D,E,F existing;
    class G,H,I,J article;
    class L missing;
    class K pending;
```

Both branches use the same cleaned records, original predictors, fixed outer partition and holdout. The original models/results were preserved. The article full GBM was fitted separately with the same settings and reproduced the original full GBM probabilities exactly. Its selected GBM is a genuinely different model, using PAY_0, LIMIT_BAL and PAY_AMT1. The original regularized LR is also present in the article report as a reference, without article likelihood tests.

Andrei's desired architecture is a shared candidate-model layer feeding both assessment frameworks, followed by a common measurable outcome criterion, potentially material harm. That final comparison is not implemented. The current structure has separate assessment branches, but also an article selection/fitting branch: variable selection is itself part of the article's proposed procedure. The LR estimator distinction and any future category pooling would require separate attribution before claiming framework superiority. Candidate models can be shared for assessment comparisons, while models produced by a framework's selection procedure still need evaluation as that procedure's outputs.

A new common model-selection/decision rule, material-harm definition and outcome experiment remain proposals. No pooling, model refitting or research-code change was performed while preparing this diagram.