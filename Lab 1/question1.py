# @author Landon


from typing import List


def get_all_subsets(some_set: List[int]) -> List[List[int]]:
    """
    1)	Given a collection of integers that might contain duplicates,
    numbers, return all possible subsets. Do not include null subset.
    """
    subsets = list()

    for i, num in enumerate(some_set):

        sub = []
        sub.append(num)
        if sub not in subsets:
            subsets.append(sub[:])

        if i <= len(some_set):
            for num_r in some_set[i + 1:]:
                sub.append(num_r)
                if sub not in subsets:
                    subsets.append(sub[:])

    subsets.sort()
    return subsets


def is_matching_set(x: List[List[int]], y: List[List[int]]) -> bool:
    """
    For testing purposes
    """
    for subset in x:
        if subset not in y:
            return False
    return True


if __name__ == "__main__":

    input_set = [1, 2, 2]
    expected_set = [[1], [2], [1, 2], [2, 2], [1, 2, 2]]
    output_set = get_all_subsets(input_set)
    print("IN: {}\nExpected: {}\nActual: {}\nPASSED?: {}".format(
        input_set, expected_set, output_set, is_matching_set(output_set, expected_set)))
