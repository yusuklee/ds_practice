import sys
import math
from collections import Counter
from itertools import combinations


def read_data(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    header = lines[0].split('\t')
    data = [line.split('\t') for line in lines[1:]]
    return header, data


def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counter = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


def gini(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counter = Counter(labels)
    return 1.0 - sum((c / total) ** 2 for c in counter.values())


def majority_class(data, class_idx):
    return Counter(row[class_idx] for row in data).most_common(1)[0][0]


def best_binary_split(data, attr_idx, class_idx):

    values = list(set(row[attr_idx] for row in data))
    n = len(values)
    if n <= 1:
        return None, -1.0

    total = len(data)
    parent_gini = gini([row[class_idx] for row in data])

    groups = {}
    for row in data:
        val = row[attr_idx]
        if val not in groups:
            groups[val] = []
        groups[val].append(row[class_idx])

    best_score = -1.0
    best_left = None

    for size in range(1, (n // 2) + 1):
        for left_combo in combinations(values, size):
            left_labels = []
            for v in left_combo:
                left_labels.extend(groups[v])
            right_labels = []
            for v in values:
                if v not in left_combo:
                    right_labels.extend(groups[v])

            if not left_labels or not right_labels:
                continue

            weighted = (len(left_labels) / total) * gini(left_labels) + \
                       (len(right_labels) / total) * gini(right_labels)
            score = parent_gini - weighted

            if score > best_score:
                best_score = score
                best_left = frozenset(left_combo)

    return best_left, best_score


def multiway_gini_score(data, attr_idx, class_idx):

    total = len(data)
    parent_gini = gini([row[class_idx] for row in data])

    partitions = {}
    for row in data:
        val = row[attr_idx]
        if val not in partitions:
            partitions[val] = []
        partitions[val].append(row[class_idx])

    weighted = sum((len(s) / total) * gini(s) for s in partitions.values())
    return parent_gini - weighted


# Threshold: binary split if unique values <= this, otherwise multi-way
BINARY_THRESHOLD = 10


def build_tree(data, class_idx, available_attrs):
    labels = [row[class_idx] for row in data]

    if len(set(labels)) == 1:
        return labels[0]

    if not available_attrs:
        return majority_class(data, class_idx)

    best_attr = None
    best_score = -1.0
    best_mode = None  # 'binary' or 'multiway'
    best_left = None

    for attr_idx in available_attrs:
        n_values = len(set(row[attr_idx] for row in data))

        if n_values <= BINARY_THRESHOLD:
            left_set, score = best_binary_split(data, attr_idx, class_idx)
            if left_set is not None and score > best_score:
                best_score = score
                best_attr = attr_idx
                best_mode = 'binary'
                best_left = left_set
        else:
            score = multiway_gini_score(data, attr_idx, class_idx)
            if score > best_score:
                best_score = score
                best_attr = attr_idx
                best_mode = 'multiway'

    if best_score <= 0 or best_attr is None:
        return majority_class(data, class_idx)

    default = majority_class(data, class_idx)

    if best_mode == 'binary':
        left_data = [row for row in data if row[best_attr] in best_left]
        right_data = [row for row in data if row[best_attr] not in best_left]
        right_vals = frozenset(set(row[best_attr] for row in data) - best_left)

        left_attrs = [a for a in available_attrs if len(set(row[a] for row in left_data)) > 1]
        right_attrs = [a for a in available_attrs if len(set(row[a] for row in right_data)) > 1]

        return {
            'type': 'binary',
            'attr': best_attr,
            'left_vals': best_left,
            'right_vals': right_vals,
            'left': build_tree(left_data, class_idx, left_attrs),
            'right': build_tree(right_data, class_idx, right_attrs),
            'default': default
        }
    else:
        partitions = {}
        for row in data:
            val = row[best_attr]
            if val not in partitions:
                partitions[val] = []
            partitions[val].append(row)

        remaining = [a for a in available_attrs if a != best_attr]
        children = {}
        for val, subset in partitions.items():
            children[val] = build_tree(subset, class_idx, remaining)

        return {
            'type': 'multiway',
            'attr': best_attr,
            'children': children,
            'default': default
        }


def classify(tree, row):
    if isinstance(tree, str):
        return tree

    if tree['type'] == 'binary':
        val = row[tree['attr']]
        if val in tree['left_vals']:
            return classify(tree['left'], row)
        elif val in tree['right_vals']:
            return classify(tree['right'], row)
        return tree['default']
    else:
        val = row[tree['attr']]
        if val in tree['children']:
            return classify(tree['children'][val], row)
        return tree['default']


def main():
    if len(sys.argv) != 4:
        print("Usage: python dt.py <train_file> <test_file> <output_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    train_header, train_data = read_data(train_file)
    class_idx = len(train_header) - 1
    class_name = train_header[class_idx]
    attributes = list(range(class_idx))

    tree = build_tree(train_data, class_idx, attributes)

    test_header, test_data = read_data(test_file)

    with open(output_file, 'w') as f:
        f.write('\t'.join(test_header + [class_name]) + '\n')
        for row in test_data:
            prediction = classify(tree, row)
            f.write('\t'.join(row + [prediction]) + '\n')


if __name__ == '__main__':
    main()
