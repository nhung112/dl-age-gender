AGE_LABELS = [
    "1-3",
    "4-12",
    "13-19",
    "20-29",
    "30-39",
    "40-59",
    "60+",
]

NUM_AGE_CLASSES = len(AGE_LABELS)

def age_to_class(age):
    if age <= 3:
        return 0
    if age <= 12:
        return 1
    if age <= 19:
        return 2
    if age <= 29:
        return 3
    if age <= 39:
        return 4
    if age <= 59:
        return 5
    return 6