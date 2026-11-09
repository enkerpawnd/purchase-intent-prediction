"""Named raw-input contracts for the matched feature-availability study.

These lists select columns only. They do not fit preprocessing or establish
that the corresponding measurements are available before a purchase.
"""

ALL_17 = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

WITHOUT_PAGEVALUES_16 = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

WITHOUT_ANALYTICS_14 = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_SETS = {
    "all_17": ALL_17,
    "without_pagevalues_16": WITHOUT_PAGEVALUES_16,
    "without_analytics_14": WITHOUT_ANALYTICS_14,
}

PAGE_COUNTS = ["Administrative", "Informational", "ProductRelated"]
DURATIONS = [f"{name}_Duration" for name in PAGE_COUNTS]
NUMERIC = ALL_17[:10]
CATEGORICAL = ALL_17[10:]

# EDA groups Weekend with discrete fields above; model preprocessing handles
# it as a binary indicator, separately from these nominal categories.
NOMINAL_CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"
]
CODE_CATEGORICAL = ["OperatingSystems", "Browser", "Region", "TrafficType"]
BOOLEAN = ["Weekend"]
